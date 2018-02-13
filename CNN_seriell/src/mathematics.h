/*
 * mathematics.h
 *
 *  Created on: 05.12.2017
 *      Author: Florian
 */

#ifndef MATHEMATICS_H_
#define MATHEMATICS_H_



namespace mathematics {

float sigmoid_once(float in);
float sigmoid_derived(float in);

void sigmoid(float *in, float *out, int size);
void sigmoid(float *in, float *out, float *derivatives, int size);


void softmax(float *in, float *out, int size);
float cross_entropy(float *calculated, float *expected, int size);

float get_cost(float *output, float *labels, int size);
float get_cost_derivatives(float *output, float *labels, float *derivatives, int size);

}


#endif /* MATHEMATICS_H_ */
