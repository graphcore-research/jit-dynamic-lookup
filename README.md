:red_circle: :warning: **Experimental and non-official Graphcore product** :warning: :red_circle:

# JIT Dynamic Lookup (JDL)

Given a 1D tensor called `data` that lives striped over many tiles on a single IPU, and an index `i` that is computed at runtime and lives on some other tile T on the same IPU, it is generally hard to have tile T access `data[i: i + N]` (for fixed `N`) because this could require dynamically exchanging data from a source tile only known at run time, and IPU exchange programs are compiled Ahead-Of-Time.

This prototype op makes Just-In-Time modifications to an Ahead-Of-Time compiled exchange program to allow this style of dynamic lookup with minimal overheads (subject to several caveats).

### Building and Running (SDK 3.2, Ubuntu 20.04, Python 3.8)

```bash
# Activate the Poplar SDK, then build the example
make

# Run the example
./example
```

The example will stripe a data tensor of integers over several tiles, then use JDL to perform several dynamic lookups and print the result.

### Caveats
 - This op only supports a single IPU.
 - This prototype doesn't support the receiving tile also being a sending tile. (i.e. the `result` tensor cannot live on a tile that also contains some of `data` tensor). This functionality would be easy to add if required, btu it wasn't required for my use case.
 - Fetching slices that straddle multiple tiles is not currently supported. Invalid indices (outside the array or straddling a tile boundary) will silently return garbage data.
 - The op is split into two programs, an expensive (~1500 cycles) planning program that only needs to be performed once during initialisations, and a cheap (~300 cycles) execution program that can be executed everytime the lookup must be performed. The planning does *not* need redoing when the lookup index changes.
 - To profile the example application, first disable printing of the tensors by commmenting out `#define DOPRINT 1` in example.cpp