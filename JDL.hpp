// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#pragma once

#include <poplar/Program.hpp>
#include <poplar/SyncType.hpp>
#include <poputil/TileMapping.hpp>


using namespace poplar;

namespace JDL {



  struct Programs {
    program::Execute setup;
    program::Sequence exchange;
  };


  Programs createPrograms(
      Graph &graph,
      const Tensor &data,
      const Tensor &tileSelector,
      const Tensor &elementSelector,
      const Tensor &result
  ) {
  /*
    Creates programs to perform a JIT Dynamic Lookup (JDL)
      ** THIS IS A PROTOTYPE **

    Args:
      - Graph &graph            : The graph to add the JDL operation to.
      - Tensor &data            : The tensor of data to slice from. Must be completely mapped.
      - Tensor &tileSelector    : A scalar INT tensor that controls which tile sends the result. This 
                                    must be the logical tileID (e.g. if `data` is mapped over tiles 20 
                                    to 50, tileSelector must take values from 20 to 50).
      - Tensor &elementSelector : A scalar INT tensor that controls the starting element to fetch from the
                                    selected tile.
      - Tensor &result          : A tensor for the output of the op. Must be completely mapped to a single
                                   tile. The size of this tensor determines the number of elements that will
                                   be fetched during the op.

    Returns `JDL::Programs progs`, where:
      - progs.setup       : is a program that should be executed exactly once to plan the exchange
      - progs.exchange    : is the program you run every time you want the planned exchange to execute

    Notes:
    - The mappings of `data` and `result` determine which tiles are senders / receivers, so they must 
      have complete mappings.
    - This prototype doesn't support the receiving tile also being a sending tile. (i.e. `result` 
      cannot live on a tile that also contains some of `data`). This would be possible to fix if required.
    - This op only supports a single IPU.
  */

    assert(graph.getTarget().getNumIPUs() == 1);
    assert(tileSelector.numElements() == 1);
    assert(elementSelector.numElements() == 1);

    // -- Parse the mappings of the input tensors -- //

    bool isMapped;
    auto dataMapping = graph.getTileMapping(data, &isMapped);
    assert(isMapped);
    auto resultMapping = graph.getTileMapping(result, &isMapped);
    assert(isMapped);
    const unsigned numTiles = graph.getTarget().getNumTiles();
    unsigned receiverTileId, numDataTiles = 0;
    for (unsigned tile = 0; tile < numTiles; ++tile) {
      int numDataIntervalsOnTile = dataMapping[tile].size();
      // Keep it simple, only support fetching from a single contiguous region
      assert(numDataIntervalsOnTile <= 1);
      numDataTiles += numDataIntervalsOnTile;
      if (!resultMapping[tile].empty()) {
        receiverTileId = tile;
        // Don't currently support sending and receiving from the same tile
        // (Would be easy to add support for this if it is needed)
        assert(numDataIntervalsOnTile == 0);
      }
    }

    // -- Graph components for the op -- //

    graph.addCodelets("JDL.gp");
    const int planSize = 9;
    unsigned numActiveTiles = numDataTiles + 1; // senders of data, plus 1 receiver
    Tensor planBuf = graph.addVariable(UNSIGNED_INT, {numActiveTiles, planSize}, "JDL_planBuf");
    Tensor dummy = graph.addVariable(UNSIGNED_INT, {numActiveTiles, 1}, "JDL_dummy");
    Tensor receiverIdConst = graph.addConstant<int>(INT, {}, receiverTileId, "JDL_receiverTileIdConst");
    Tensor countConst = graph.addConstant<unsigned>(UNSIGNED_INT, {}, result.numElements(), "JDL_countConst");
    ComputeSet setupCS = graph.addComputeSet("JDL_setupCS"); 
    ComputeSet exchangeCS = graph.addComputeSet("JDL_exchangeCS");

    // -- First setup the single receiver tile -- //

    VertexRef setupVtx = graph.addVertex(setupCS, "JDLSetupRecv", {
      {"planBuf", planBuf[numActiveTiles - 1]},
      {"count", countConst}
    });
    VertexRef exchangeVtx = graph.addVertex(exchangeCS, "JDLRecv", {
      {"planBuf", planBuf[numActiveTiles - 1]},
      {"nonexecutableDummy", dummy[numActiveTiles - 1]},
      {"tileSelector", tileSelector},
      {"result", result},
    });
    graph.setTileMapping(setupVtx, receiverTileId);
    graph.setTileMapping(exchangeVtx, receiverTileId);
    graph.setTileMapping(planBuf[numActiveTiles-1], receiverTileId);
    graph.setTileMapping(dummy[numActiveTiles-1], receiverTileId);
    graph.setTileMapping(receiverIdConst, receiverTileId);
    graph.setTileMapping(countConst, receiverTileId);

    // -- Next setup the sender tiles (the ones who have data) -- //

    for (unsigned tile = 0, dataTile = 0; tile < numTiles; ++tile) {
      if (tile == receiverTileId) {
        // Receiver tile is ignored
        continue; 
      }
      if (dataMapping[tile].size() == 0) {
        // Inactive tiles  must signal non-participation
        VertexRef vtx = graph.addVertex(exchangeCS, "JDLNonParticipationVtx");
        graph.setTileMapping(vtx, tile);
        continue;
      }
      // Otherwise, set up sender tile
      setupVtx = graph.addVertex(setupCS, "JDLSetupSend", {
        {"planBuf", planBuf[dataTile]},
        {"recvTile", receiverIdConst},
        {"count", countConst}
      });
      exchangeVtx = graph.addVertex(exchangeCS, "JDLSend", {
          {"planBuf", planBuf[dataTile]},
          {"nonexecutableDummy", dummy[dataTile]},
          {"elementSelector", elementSelector},
          {"data", data[dataTile]},
        });
      graph.setTileMapping(setupVtx, tile);
      graph.setTileMapping(exchangeVtx, tile);
      graph.setTileMapping(planBuf[dataTile], tile);
      graph.setTileMapping(dummy[dataTile], tile);

      dataTile++;
    }

    // -- Make the output programs -- //
    program::Execute setupProgram(setupCS);
    program::Sequence exchangeProgram({
      program::Sync(SyncType::INTERNAL), // Hack to make poplar sync analysis work, costs time
      program::Execute(exchangeCS),
      program::Sync(SyncType::INTERNAL),
    });

    return {setupProgram, exchangeProgram};
  }



}