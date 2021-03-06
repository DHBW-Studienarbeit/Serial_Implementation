/*
 * matrix.cpp
 *
 *    n
 *  +-----+
 * m|x x x|
 *  |x x x|
 *  +-----+
 *
 *  Created on: 29.11.2017
 *      Author: Josua Benz
 */

#include "matrix.hpp"
#include <stdlib.h>
#include <iostream>

/**
 * Constructor for Class Matrix with no Values for Matrix
 * <param>int length - length of Matrix</param>
 * <param>int height - height of Matrix</param>
 */
Matrix::Matrix(int height, int length){
	this->mat_array=new float[length*height];
	this->length=length; //n
	this->height=height; //m
}

/**
 * Constructor for Class Matrix with Values for Matrix
 * Length of Array has to be height*length
 * <param>int length - length of Matrix</param>
 * <param>int height - height of Matrix</param>
 * <param>float *array - Pointer to Array with Values
 */
Matrix::Matrix(int height, int length, float *array){
	this->mat_array=array;
	this->length=length; //n
	this->height=height; //m
}

/**
 * Default Destructor of Class Matrix
 */
Matrix::~Matrix(){
	delete[] mat_array;
}

void Matrix::set_all(float* array)
{
	delete[] mat_array;
	mat_array = array;
}

void Matrix::copy_all(float* array)
{
	int i = 0;
	for(int m=0; m<height; m++){
		for(int n=0; n<length; n++, i++){
			set(m,n,array[i]);
		}
	}
}

void Matrix::set_all_equal(float value)
{
	for(int m=0; m<height; m++){
		for(int n=0; n<length; n++){
			set(m,n,value);
		}
	}
}

/**
 * Function to get Value on Position (n,m)
 * value range of n & m between 0 and height/length-1
 * <param>int n - Column-Index</param>
 * <param>int m - Row-Index</param>
 * <return>float - Value at position in Matrix</return>
 */
float Matrix::get(int m, int n){
	if (m < height && n < length)
	{
		return this->mat_array[m*(this->length) + n];
	}
	else
	{
		std::cerr << "invalid arguments in matrix::get m:" << m << " n:" << n;
		return 0.0f;
	}
}

/*
 * Function to get a Pointer to the entire Array
 * <return>float * - Pointer to Array with length height*length</return>
 */
float* Matrix::get(){
	return mat_array;
}

/**
 * Function to get Height of Matrix
 * <return>int - Height of Matrix</return>
 */
int Matrix::getHeight(){
	return height;
}

/**
 * Function to get Length of Matrix
 * <return>int - Length of MAtrix</return>
 */
int Matrix::getLength(){
	return length;
}

/**
 * Function to set the Value on an specific Point in the Matrix
 * value range of n & m between 0 and height/length-1
 * <param>int n - Column-Index</param>
 * <param>int m - Row-Index</param>
 * <param>float value - Value to be set</param>
 */
void Matrix::set(int m, int n, float value){
	if (m < height && n < length)
	{
		this->mat_array[m*(this->length) + n] = value;
	}
	else
	{
		std::cerr << "invalid arguments in matrix::set m:" << m << " n:" << n;
	}
}

/**
 * Function to print out Matrix on Console
 */
void Matrix::printOut(){
	for(int m=0; m<height; m++){
		std::cout << std::endl;
		for(int n=0; n<length; n++){
				std::cout << get(m,n) << "  ";
		}
	}
	std::cout << std::endl << std::endl;
}

/**
 * Function to fill Matrix with Test-Values.
 * Value equals the Colums-Index
 */
void Matrix::test(){
	for(int m=0; m<height; m++){
		for(int n=0; n<length; n++){

			//generate random number on position [m][n]
			set(m,n, (float)n);
		}
	}
}

/**
 * Function to fill the Matrix with Test-Values between 0 and 1
 */
void Matrix::random(){
	for(int m=0; m<height; m++){
		for(int n=0; n<length; n++){

			//generate random number on position [m][n]
			set(m ,n , static_cast <float> (rand()) / (static_cast <float> (RAND_MAX)));
		}
	}
}

/**
 * Function to transpose the Matrix
 */
void Matrix::trans(){

	//Matrix *new_matrix = new Matrix(this->height , this->length);
	float *new_data=new float[this->height * this->length];
	for(int m=0; m<height; m++){
		for(int n=0; n<length; n++){
			//addition of both arguments
			new_data[m*height+n]=get(m,n);
		}
	}
	delete mat_array;
	mat_array = new_data;
	int tmp = height;
	height = length;
	length = tmp;
}

/**
 * Operator-Function for addition of two Matrixes
 * Matrixes have to be the same size
 * <param>Matrix &a - first Matrix</param>
 * <param>Matrix &b - second Matrix</param>
 * <return> Matrix </return>
 */
Matrix operator+ (Matrix &a, Matrix &b){
	Matrix *c = new Matrix(a.getHeight(), a.getLength());
	if(a.getLength() == b.getLength() && a.getHeight() == b.getHeight()){
		for(int m=0; m<c->getHeight(); m++){
			for(int n=0; n<c->getLength(); n++){
				//addition of both arguments
				c->set(m, n, a.get(m,n) + b.get(m, n));
			}
		}
		return *c;
	}else{
		std::cerr << "invalid arguments for matrix addition";
		return *c;
	}
}

/**
 * Operator-Function for subtraction of two Matrixes
 * Matrixes have to be the same size
 * <param>Matrix &a - first Matrix</param>
 * <param>Matrix &b - second Matrix</param>
 * <return> Matrix </return>
 */
Matrix operator- (Matrix &a, Matrix &b){
	Matrix *c = new Matrix(a.getHeight(), a.getLength());
	if(a.getLength() == b.getLength() && a.getHeight() == b.getHeight()){
		for(int m=0; m<c->getHeight(); m++){
			for(int n=0; n<c->getLength(); n++){
				//addition of both arguments
				c->set(m, n, a.get(m,n) - b.get(m, n));
			}
		}
		return *c;
	}else{
		std::cerr << "\ninvalid arguments for matrix subtraction";
		return *c;
	}
}

/**
 * Operator-Function for multiplication of an scalar and a Matrix
 * Matrixes have to be the same size
 * <param>int a - scalar</param>
 * <param>Matrix &b - Matrix</param>
 * <return> Matrix </return>
 */
Matrix operator* (int a, Matrix &b){
	Matrix *c = new Matrix(b.getHeight(), b.getLength());
	for(int m=0; m < c->getHeight(); m++){
		for(int n=0; n < c->getLength(); n++){
			//addition of both arguments
			c->set(m, n, a * b.get(m,n) );
		}
	}
	return *c;
}

/* Operator-Function for multiplication of 2 Matrixes
 * Length of first Matrix has to be the same as Height of second Matrix
 * Return Matrix has size of the Height of first - and Length of second Matrix
 *
 *  n
 * +--+
 *m|  |
 * +--+
 *
 * +-----+               +-------+
 * |x x x|   +-------+   |x x x x|
 * |x x x|   |x x x x|   |x x x x|
 * |x x x| x |x x x x| = |x x x x|
 * |x x x|   |x x x x|   |x x x x|
 * |x x x|   +-------+   |x x x x|
 * +-----+               +-------+
 *
 * <param>Matrix &a - first Matrix</param>
 * <param>Matrix &b - second Matrix</param>
 * <return> Matrix </return>
 *
 */
Matrix operator* (Matrix &a, Matrix &b){
	Matrix *c = new Matrix(a.getHeight(), b.getLength());
	if(a.getLength() == b.getHeight()){

		//Alle Elemente der neuen Matrix durchgehen
		for(int m=0; m < c->getHeight(); m++){
			for(int n=0; n < c->getLength(); n++){

				float value=0;
				//Zeilenweise durchgehen
				for(int i=0; i<a.getLength(); i++){
					value += (a.get(m, i) * b.get(i, n));
				}
				c->set(m, n, value);

			}
		}
		return *c;

	}else{
		std::cerr << "\ninvalid arguments for matrix multiplication";
		return *c;
	}
}
