cmake_minimum_required(VERSION 3.10)
project(Lenet5 VERSION 1.0.0 LANGUAGES CXX CUDA)
find_package(CUDA)

set_source_files_properties(ConvNet.cpp CNN.cpp ConvLayer.cpp SamplingLayer.cpp FullyConnectedLayer.cpp OutputLayer.cpp PROPERTIES LANGUAGE CUDA)
add_library(
	CNN
	CNN.h
	CNN.cpp
	)

add_library(
	ConvLayer
	ConvLayer.h
	ConvLayer.cpp
	)

add_library(
	SamplingLayer
	SamplingLayer.h
	SamplingLayer.cpp
	)

add_library(
	FullyConnectedLayer
	FullyConnectedLayer.h
	FullyConnectedLayer.cpp
	)

add_library(
	OutputLayer
	OutputLayer.h
	OutputLayer.cpp
	)

add_executable(Lenet5_CUDA ConvNet.cpp)
target_compile_options(Lenet5_CUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_30 -std=c++14>)
set_property(TARGET Lenet5_CUDA PROPERTY CUDA_RESOLVE_SYMBOLS ON)
target_link_libraries(Lenet5_CUDA
	PRIVATE CNN
	PRIVATE ConvLayer
	PRIVATE SamplingLayer
	PRIVATE FullyConnectedLayer
	PRIVATE OutputLayer
	PRIVATE -lcublas
	)
