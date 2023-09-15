// Copyright (c) 2023 Graphcore Ltd. All rights reserved.


// Problem definition
const unsigned numDataTiles = 5;
const unsigned numElementsPerDataTile = 10;
const unsigned lookupSize = 3;
const unsigned receiverTileId = 1286;  // Just some random tile



#ifndef __IPU__
// CPU code goes here

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <popops/Loop.hpp>

#include "JDL.hpp"

using namespace poplar;


#define DOPRINT (1)


int main() {

    // Setup device and graph
    auto devManager = DeviceManager();
    auto devs = devManager.getDevices(TargetType::IPU, 1);
    Device &device = devs[0];
    device.attach();
    Target target = device.getTarget();
    Graph graph(target);


    // Create initial data, split over multiple tiles
    srand(0);
    std::vector<int> data_h(numDataTiles * numElementsPerDataTile);
    for (unsigned i = 0; i < numDataTiles * numElementsPerDataTile; ++i) {
        data_h[i] = rand() % 100;
    }
    Tensor data = graph.addVariable(INT, {numDataTiles, numElementsPerDataTile}, "data");
    graph.setInitialValue<int>(data, data_h);
    for (unsigned tile = 0; tile < numDataTiles; ++tile) {
      graph.setTileMapping(data[tile], tile);
    }

    // Create the JDL programs
    Tensor tileSelector = graph.addVariable(INT, {}, "tileSelector");
    Tensor elementSelector = graph.addVariable(INT, {}, "elementSelector");
    Tensor result = graph.addVariable(INT, {lookupSize}, "result");
    graph.setTileMapping(tileSelector, receiverTileId); 
    graph.setTileMapping(elementSelector, receiverTileId); 
    graph.setTileMapping(result, receiverTileId);

    JDL::Programs jdlProgams = JDL::createPrograms(graph, data, tileSelector, elementSelector, result);


    // Some example user code that generates requests for elements
    graph.addCodelets(__FILE__);
    ComputeSet requestCS = graph.addComputeSet("requestCS");
    VertexRef vtx = graph.addVertex(requestCS, "RequestGenerator", {
      {"elementSelector", elementSelector},
      {"tileSelector", tileSelector},
    });
    graph.setTileMapping(vtx, receiverTileId);


    // A program that sets up the exchange once, then generates several requests and executes them
    program::Sequence mainProgram({
#ifdef DOPRINT
      program::PrintTensor("Data", data),
#endif
      jdlProgams.setup,
      program::Repeat(5, program::Sequence({
        program::Execute(requestCS),
        jdlProgams.exchange,
#ifdef DOPRINT
        program::PrintTensor("\nTile Selected", tileSelector),
        program::PrintTensor("Element Selected", elementSelector),
        program::PrintTensor("Result", result),
#endif
      })),
    });

    // Run
    Engine engine(graph, mainProgram);
    engine.load(device);
    engine.run(0); 
  
    return EXIT_SUCCESS;
}


#else 
// IPU code goes here


#include <poplar/Vertex.hpp>

// An simple example codelet that generates random requests
struct RequestGenerator : public poplar::Vertex {
    poplar::Output<int> elementSelector;
    poplar::Output<int> tileSelector;
    void compute() {   
        *tileSelector = __builtin_ipu_urand32() % numDataTiles;
        *elementSelector = __builtin_ipu_urand32() % (numElementsPerDataTile - lookupSize + 1);
    }
};


#endif 