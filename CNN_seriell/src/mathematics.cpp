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

float inverse_sigmoid_once(float in)
{
	return log(in / (1+in));
}

void sigmoid(float *in, float *out, int size)
{
	for(; size>0; size--, in++, out++)
	{
		*out = sigmoid_once(*in);
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


} /* namespace mathematics */
