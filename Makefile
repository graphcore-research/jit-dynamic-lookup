# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

example: example.cpp JDL.hpp
	g++ $< -std=c++17 -lpoplar -lpopops -lpoputil -Wall -Wextra -o $@

clean:
	rm example
