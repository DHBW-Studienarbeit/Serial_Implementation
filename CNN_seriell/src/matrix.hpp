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
	Matrix(int height, int length);
	Matrix(int height, int length, float *array);
	~Matrix();
	void random();
	void test();
	void trans();
	void printOut();
	int getLength();
	int getHeight();
	float* get();
	float get(int m,int n);
	void set(int m, int n, float value);
	void set_all_equal(float value);
	void copy_all(float* array);
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
