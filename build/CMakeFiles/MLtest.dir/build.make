# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.7.1/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.7.1/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/cypress/Works/MachineLearning

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/cypress/Works/MachineLearning/build

# Include any dependencies generated for this target.
include CMakeFiles/MLtest.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/MLtest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MLtest.dir/flags.make

CMakeFiles/MLtest.dir/BP.cpp.o: CMakeFiles/MLtest.dir/flags.make
CMakeFiles/MLtest.dir/BP.cpp.o: ../BP.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/cypress/Works/MachineLearning/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MLtest.dir/BP.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MLtest.dir/BP.cpp.o -c /Users/cypress/Works/MachineLearning/BP.cpp

CMakeFiles/MLtest.dir/BP.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MLtest.dir/BP.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cypress/Works/MachineLearning/BP.cpp > CMakeFiles/MLtest.dir/BP.cpp.i

CMakeFiles/MLtest.dir/BP.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MLtest.dir/BP.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cypress/Works/MachineLearning/BP.cpp -o CMakeFiles/MLtest.dir/BP.cpp.s

CMakeFiles/MLtest.dir/BP.cpp.o.requires:

.PHONY : CMakeFiles/MLtest.dir/BP.cpp.o.requires

CMakeFiles/MLtest.dir/BP.cpp.o.provides: CMakeFiles/MLtest.dir/BP.cpp.o.requires
	$(MAKE) -f CMakeFiles/MLtest.dir/build.make CMakeFiles/MLtest.dir/BP.cpp.o.provides.build
.PHONY : CMakeFiles/MLtest.dir/BP.cpp.o.provides

CMakeFiles/MLtest.dir/BP.cpp.o.provides.build: CMakeFiles/MLtest.dir/BP.cpp.o


CMakeFiles/MLtest.dir/MFNN.cpp.o: CMakeFiles/MLtest.dir/flags.make
CMakeFiles/MLtest.dir/MFNN.cpp.o: ../MFNN.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/cypress/Works/MachineLearning/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/MLtest.dir/MFNN.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MLtest.dir/MFNN.cpp.o -c /Users/cypress/Works/MachineLearning/MFNN.cpp

CMakeFiles/MLtest.dir/MFNN.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MLtest.dir/MFNN.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/cypress/Works/MachineLearning/MFNN.cpp > CMakeFiles/MLtest.dir/MFNN.cpp.i

CMakeFiles/MLtest.dir/MFNN.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MLtest.dir/MFNN.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/cypress/Works/MachineLearning/MFNN.cpp -o CMakeFiles/MLtest.dir/MFNN.cpp.s

CMakeFiles/MLtest.dir/MFNN.cpp.o.requires:

.PHONY : CMakeFiles/MLtest.dir/MFNN.cpp.o.requires

CMakeFiles/MLtest.dir/MFNN.cpp.o.provides: CMakeFiles/MLtest.dir/MFNN.cpp.o.requires
	$(MAKE) -f CMakeFiles/MLtest.dir/build.make CMakeFiles/MLtest.dir/MFNN.cpp.o.provides.build
.PHONY : CMakeFiles/MLtest.dir/MFNN.cpp.o.provides

CMakeFiles/MLtest.dir/MFNN.cpp.o.provides.build: CMakeFiles/MLtest.dir/MFNN.cpp.o


# Object files for target MLtest
MLtest_OBJECTS = \
"CMakeFiles/MLtest.dir/BP.cpp.o" \
"CMakeFiles/MLtest.dir/MFNN.cpp.o"

# External object files for target MLtest
MLtest_EXTERNAL_OBJECTS =

MLtest: CMakeFiles/MLtest.dir/BP.cpp.o
MLtest: CMakeFiles/MLtest.dir/MFNN.cpp.o
MLtest: CMakeFiles/MLtest.dir/build.make
MLtest: CMakeFiles/MLtest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/cypress/Works/MachineLearning/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable MLtest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MLtest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MLtest.dir/build: MLtest

.PHONY : CMakeFiles/MLtest.dir/build

CMakeFiles/MLtest.dir/requires: CMakeFiles/MLtest.dir/BP.cpp.o.requires
CMakeFiles/MLtest.dir/requires: CMakeFiles/MLtest.dir/MFNN.cpp.o.requires

.PHONY : CMakeFiles/MLtest.dir/requires

CMakeFiles/MLtest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MLtest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MLtest.dir/clean

CMakeFiles/MLtest.dir/depend:
	cd /Users/cypress/Works/MachineLearning/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/cypress/Works/MachineLearning /Users/cypress/Works/MachineLearning /Users/cypress/Works/MachineLearning/build /Users/cypress/Works/MachineLearning/build /Users/cypress/Works/MachineLearning/build/CMakeFiles/MLtest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MLtest.dir/depend

