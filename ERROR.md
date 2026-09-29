# C to C++ Transition Audit Report

The project has been successfully migrated to C++ in most aspects. You've correctly replaced raw arrays with `std::vector`, used `std::string`, leveraged C++ stream I/O (`<iostream>`, `<fstream>`, `<sstream>`), and moved away from C-style structs to classes and namespaces.

However, there is **one major error** related to the transition of random number generation that breaks program correctness (reproducibility):

## 1. Random Number Generator Seeding is Broken
In `la1.cpp` (lines 235-241), you are using the old C-style `std::srand(seeds[i])` from `<cstdlib>` to seed the random number generator for different iterations:

```cpp
        for (size_t i = 0; i < seeds.size(); ++i)
        {
            // ...
            std::srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset, testDataset, iterations, trainErrors[i], testErrors[i]);
            // ...
        }
```

However, you updated `util.cpp` (`util::randomInt` and `util::randomDouble`) and `util.h` (`util::integerRandomVectorWithoutRepeating`) to use the modern C++11 `<random>` library, initializing it dynamically with `std::random_device`:

```cpp
    static std::random_device rd;
    static std::mt19937 gen(rd());
```

**The problem**: `std::mt19937` does not look at the seed set by `std::srand`. Furthermore, because `gen` is defined as `static` and initialized with `rd()`, the random engine is initialized exactly once per program execution with a non-deterministic seed. This means:
1. The 5 seeds in `la1.cpp` will actually use the same underlying random sequence, because the `static` generator is not being re-seeded.
2. The results will not be reproducible across different runs of the program, completely breaking the purpose of using fixed seeds (`{1, 2, 3, 4, 5}`).

### Recommendation
You should expose a seeding function in your `util` namespace that properly seeds the `std::mt19937` engine, instead of using `std::srand()`.

For example, you could have a globally accessible generator in `util` or a `setSeed` function:

```cpp
// In util.h
namespace util {
    void setSeed(int seed);
    // ...
}

// In util.cpp
static std::mt19937 gen; // Default initialized

void util::setSeed(int seed) {
    gen.seed(seed);
}

double util::randomDouble(double Low, double High) {
    std::uniform_real_distribution<double> dis(Low, High);
    return dis(gen);
}
// ...
```
Then, update `la1.cpp` to call `util::setSeed(seeds[i]);` instead of `std::srand(seeds[i]);`. You should also do this for `integerRandomVectorWithoutRepeating` in `util.h`.
