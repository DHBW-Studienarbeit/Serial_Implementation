/*
 * mathematics.cpp
 *
 *  Created on: 05.12.2017
 *      Author: Florian
 */

#include "mathematics.h"
#include "math.h"
#include <iostream>

namespace mathematics {


float sigmoid_once(float in)
{
	double temp = exp(in);
	return (float)(temp / (1+temp));
}

float sigmoid_backward_derivated_once(float activation)
{
	return activation * (1 - activation);
}

void sigmoid(float *in, float *out, int size)
{
	for(; size>0; size--, in++, out++)
	{
		*out = sigmoid_once(*in);
	}
}

void sigmoid_backward_derivated(float *activation, float *derivatives, int size)
{
	for(; size>0; size--, activation++, derivatives++)
	{
		*derivatives = sigmoid_backward_derivated_once(*activation);
	}
}


void softmax(float *in, float *out, int size)
{
	double sum=0;
	for(int i=0; i<size; i++)
	{
		sum += exp(in[i]);
	}
	for(int i=0; i<size; i++)
	{
		out[i] = (float)(exp(in[i]) / sum);
	}
}


float cross_entropy(float *calculated, float *expected, int size)
{
	double sum=0;
	for(; size>0; size--, expected++, calculated++)
	{
		sum += - (*expected) * log(*calculated);
	}
	return (float)sum;
}

float get_cost(float *output, float *labels, int size)
{
	float *normalized;
	float ret;
	normalized = new float[size];
	softmax(output, normalized, size);
	ret = cross_entropy(normalized, labels, size);
	delete[] normalized;
	return ret;
}

void get_cost_derivatives(float *output, float *labels, float *derivatives, int size)
{
	for(; size>0; size--, output++, labels++, derivatives++)
	{
		*derivatives = *output - *labels;
	}
}

/**
 * Function for FullyConnected and Conv Layer.
 *
 * <param>Matrix *weight - Weight Matrix</param>
 * <param>Matrix *node - Node Matrix, only length of 1</param>
 * <param>Matrix *bias - Bias Matrix, only lengt of 1</param>
 * <result>Matrix for Sigmoid-Function</result>
 *
 */
Matrix *conv(Matrix *weight, Matrix *node, Matrix *bias){
	if(weight->getLength() == node->getHeight() && bias->getHeight() == weight->getHeight() && bias->getLength() == 1){
		Matrix *output = new Matrix(weight->getHeight(), 1);
		#pragma omp parallel for
		for(int m=0; m < weight->getHeight(); m++){
			float value=0;
			for(int i=0; i<weight->getLength(); i++){
				value += (weight->get(m, i)*node->get(i,0));
			}
			output->set(m, 0, value+bias->get(m,0));
		}
		return output;
	}else{
		std::cerr << "invalid arguments for matrix conv" << weight->getLength() << node->getHeight() << bias->getHeight() << weight->getHeight() << bias->getLength();
		return new Matrix(0,0);
	}
}

} /* namespace mathematics */
