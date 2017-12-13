/*
 * mathematics.h
 *
 *  Created on: 05.12.2017
 *      Author: Florian
 */

#ifndef MATHEMATICS_H_
#define MATHEMATICS_H_



namespace mathematics {


void sigmoid(float *in, float *out, int size);
void softmax(float *in, float *out, int size);

float cross_entropy(float *calculated, float *expected, int size);

}


#endif /* MATHEMATICS_H_ */
