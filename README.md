:red_circle: :warning: **Experimental and non-official Graphcore product** :warning: :red_circle:

# JIT Dynamic Lookup (JDL)

Given a 1D tensor called `data` that lives striped over many tiles on a single IPU, and an index `i` that is computed at runtime and lives on some other tile `T` on the same IPU, it is generally hard to have tile `T` access `data[i: i + N]` (for fixed `N`) because this could require dynamically exchanging data from a source tile only known at run time, and IPU exchange programs are compiled Ahead-Of-Time.

This prototype op makes Just-In-Time modifications to an Ahead-Of-Time compiled exchange program to allow this style of dynamic lookup with minimal overheads (subject to several caveats).

### Building and Running (SDK 3.2, Ubuntu 20.04, Python 3.8)

```bash
# Activate the Poplar SDK, then build the example
make

# Run the example
./example
```

The example will stripe a data tensor of integers over several tiles, then use JDL to perform several dynamic lookups and print the result.

### Usage
If you have variables that are set up something like this:

```c++
int numDataTiles = 1024;
int numElementsPerDataTile = 1024;
int lookupTile = 1025; // ID of tile that gathers the output
int lookupSize = 16;

// Put the data tensor on some datatiles
Tensor data = graph.addVariable(INT, {numDataTiles, numElementsPerDataTile}, "data");
for (unsigned tile = 0; tile < numDataTiles; ++tile) {
    graph.setTileMapping(data[tile], tile);
}

// Control variables on the tile that wants to do the lookup
Tensor tileSelector = graph.addVariable(INT, {}, "tileSelector");
Tensor elementSelector = graph.addVariable(INT, {}, "elementSelector");
Tensor result = graph.addVariable(INT, {lookupSize}, "result");
graph.setTileMapping(tileSelector, lookupTile); 
graph.setTileMapping(elementSelector, lookupTile); 
graph.setTileMapping(result, lookupTile);
```

Then the API call looks like this:
```c++
JDL::Programs jdlPrograms = JDL::createPrograms(graph, data, tileSelector, elementSelector, result);
```

This builds you two `poplar::Program`s. At the start of your IPU program, execute `progs.setup` once. Later, execute `progs.exchange` as many times as you want, which will populate `result` by fetching data from `data` based on the current values stored in `tileSelector` and `elementSelector`.

```c++
program::Sequence mainProgram({

    jdlPrograms.setup,
    // ...

    program::Repeat(999, 
        program::Sequence({
            // ... put programs here that modify `tileSelector` and `elementSelector`
            jdlPrograms.exchange,
            // ... put programs here that use the output from `result`
        }
    )),
});
```
The op always fetches `result.size()` elements.


### Caveats
 - This op only supports a single IPU.
 - This prototype doesn't support the receiving tile also being a sending tile. (i.e., the `result` tensor cannot live on a tile that also contains some of `data` tensor). This functionality would be easy to add if required, but it wasn't required for my use case.
 - Fetching slices that straddle multiple tiles is not currently supported. Invalid indices (outside the array or straddling a tile boundary) will silently return garbage data.
 - The op is split into two programs, an expensive (~1500 cycles) planning program that only needs to be performed once during initialisations, and a cheap (~300 + `lookupSize` cycles) execution program that can be executed every time the lookup must be performed. The planning does *not* need redoing when the lookup index changes.
 - To profile the example application, first disable printing of the tensors by commenting out `#define DOPRINT 1` in `example.cpp`


## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License.

The included code is released under an MIT license, (see [LICENSE](LICENSE)).
