/*
 * ConvLayer.hpp
 *
 *  Created on: 29.11.2017
 *      Author: "Benjamin Riedle"
 */

#ifndef CONVLAYER_HPP_
#define CONVLAYER_HPP_

#include "Layer.hpp"

class Conv_Layer: public Layer {

private:
	int x_receptive;
	int y_receptive;
	int step_size;
	int no_feature_maps;
	int x_size;
	int y_size;
public:
	Conv_Layer(int x_receptive, int y_receptive, int step_size, int no_feature_maps);
	virtual ~Conv_Layer();
	int  getXSize();
	int  getYSize();
	void setXSize(int size);
	void setYSize(int size);
	int  getXReceptive();
	int  getYReceptive();
	int  getStepSize();
	int  getNoFeatureMaps();

	virtual void backpropagate( Matrix* inputs,
								Matrix* activations,
								Matrix* input_derivations,
								Matrix* activation_derivations,
								Matrix* weights,
								Matrix* biases,
								Matrix* weight_derivations,
								Matrix* bias_derivations ) override;
};

#endif /* CONVLAYER_HPP_ */
