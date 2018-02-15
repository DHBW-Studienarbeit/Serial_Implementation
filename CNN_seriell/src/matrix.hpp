/*
 * matrix.hpp
 *
 *  Created on: 29.11.2017
 *      Author: Josua Benz
 */

#ifndef MATRIX_HPP_
#define MATRIX_HPP_

class Matrix
{
public:
	Matrix(int length, int height);
	Matrix(int length, int height, float *array);
	~Matrix();
	void random();
	void test();
	void trans();
	void printOut();
	int getLength();
	int getHeight();
	float* get();
	float get(int n,int m);
	void set(int n, int m, float value);
	void set_all(float* array);

private:
	int length;
	int height;
	float *mat_array;
};
Matrix operator+(Matrix &a, Matrix &b);
Matrix operator-(Matrix &a, Matrix &b);
Matrix operator*(int a, Matrix &b);
Matrix operator*(Matrix &a, Matrix &b);
#endif /* MATRIX_HPP_ */
