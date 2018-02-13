/*
 * mathematics.cpp
 *
 *  Created on: 05.12.2017
 *      Author: Florian
 */

#include "mathematics.h"
#include "math.h"

namespace mathematics {


float sigmoid_once(float in)
{
	double temp = exp(in);
	return temp / (1+temp);
}

float sigmoid_derived(float in)
{
	float sig = sigmoid_once(in);
	return sig * (1 - sig);
}

void sigmoid(float *in, float *out, int size)
{
	for(; size>0; size--, in++, out++)
	{
		*out = sigmoid_once(*in);
	}
}

void sigmoid(float *in, float *out, float *derivatives, int size)
{
	for(; size>0; size--, in++, out++, derivatives++)
	{
		*out = sigmoid_once(*in);
		*derivatives = (*out) * (1 - *out);
	}
}


void softmax(float *in, float *out, int size)
{
	float sum=0;
	for(int i=0; i<size; i++)
	{
		sum += exp(in[i]);
	}
	for(int i=0; i<size; i++)
	{
		out[i] = exp(in[i]) / sum;
	}
}


float cross_entropy(float *calculated, float *expected, int size)
{
	float sum=0;
	for(; size>0; size--, expected++, calculated++)
	{
		sum += - (*expected) * log(*calculated);
	}
	return sum;
}

float get_cost(float *output, float *labels, int size)
{
	float normalized[size];
	softmax(output, normalized, size);
	return cross_entropy(normalized, labels, size);
}

float get_cost_derivatives(float *output, float *labels, float *derivatives, int size)
{
	for(; size>0; size--, output++, labels++, derivatives++)
	{
		*derivatives = *output - *labels;
	}
}

} /* namespace mathematics */
