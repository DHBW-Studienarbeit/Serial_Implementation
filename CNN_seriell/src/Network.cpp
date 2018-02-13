/*
 * network.cpp
 *
 *  Created on: 23.11.2017
 *      Author:
 *
 *  This is the implementation of the Network class
 */

#include "Network.hpp"
#include <math.h>


Network::Network()
{
	layer_list = new vector<Layer*>();
	node_list = new vector<Matrix*>();
	weight_list = new vector<Matrix*>();
	bias_list = new vector<Matrix*>();
}

Network::~Network()
{
	for(unsigned int i = 0; i < layer_list->size(); i++)
	{
		delete layer_list->at(i);
	}

	for(unsigned int i = 0; i < layer_list->size(); i++)
	{
		delete node_list->at(i);
	}

	for(unsigned int i = 0; i < layer_list->size(); i++)
	{
		delete weight_list->at(i);
	}

	for(unsigned int i = 0; i < layer_list->size(); i++)
	{
		delete bias_list->at(i);
	}

	delete layer_list;
	delete node_list;
	delete weight_list;
	delete bias_list;
}

void Network::add_Layer(Layer* layer)
{
	layer_list->push_back(layer);
}

/**
 * This function sets up all Layers contained in layer_list as matrices
 * Make sure that your network is clear before you generate a new network
 *
 * <return> bool - indicates if initialization of the network was successful.
 * It can fail if your layers are not sorted in an expected manner </return>
 */
bool Network::generate_network()
{
	for(unsigned int i = 0; i < layer_list->size(); i++)
	{
		Layer* layer = layer_list->at(i);
		switch(layer->getLayerType())
		{
			case INPUT_LAYER:
			{
				Input_Layer* input_layer = (Input_Layer*) layer;
				node_list->push_back(new Matrix(input_layer->getRows(),input_layer->getCols()));
				break;
			}
			case CONV_LAYER:
			{
				Conv_Layer* conv_layer = (Conv_Layer*) layer;
				if((layer_list->at(i-1)->getLayerType() == INPUT_LAYER))
				{
					Input_Layer* last_layer = (Input_Layer*) layer_list->at(i-1);
					int prev_dim_x = last_layer->getCols();
					int prev_dim_y = last_layer->getRows();

					int dim_x = (prev_dim_x - conv_layer->getXReceptive() + 1) / conv_layer->getStepSize();
					int dim_y = (prev_dim_y - conv_layer->getYReceptive() + 1) / conv_layer->getStepSize();

					conv_layer->setXSize(dim_x);
					conv_layer->setYSize(dim_y);

					for (int i = 0; i < conv_layer->getNoFeatureMaps(); i++)
					{
						node_list->push_back(new Matrix(dim_y, dim_x));
						weight_list->push_back(new Matrix(conv_layer->getXSize(), conv_layer->getYSize()));
						bias_list->push_back(new Matrix(dim_y, dim_x));
					}
					conv_layer->setSize(conv_layer->getNoFeatureMaps()*dim_x*dim_y);
				}
				else if((layer_list->at(i-1)->getLayerType() == POOLING_LAYER))
				{
					MaxPooling_Layer* last_layer = (MaxPooling_Layer*) (layer_list->at(i-1));

					int prev_dim_x = last_layer->getXSize();
					int prev_dim_y = last_layer->getYSize();

					int dim_x = (prev_dim_x - conv_layer->getXReceptive() + 1) / conv_layer->getStepSize();
					int dim_y = (prev_dim_y - conv_layer->getYReceptive() + 1) / conv_layer->getStepSize();

					conv_layer->setXSize(dim_x);
					conv_layer->setYSize(dim_y);

					for (int i = 0; i < conv_layer->getNoFeatureMaps(); i++)
					{
						node_list->push_back(new Matrix(dim_y, dim_x));
						weight_list->push_back(new Matrix(conv_layer->getXSize(), conv_layer->getYSize()));
						bias_list->push_back(new Matrix(dim_y, dim_x));
					}

					conv_layer->setSize(conv_layer->getNoFeatureMaps()*dim_x*dim_y);
				}
				else
				{
					return false;
				}
				break;
			}
			case POOLING_LAYER:
			{
				MaxPooling_Layer* pooling_layer = (MaxPooling_Layer*) layer;
				if((layer_list->at(i-1)->getLayerType() == CONV_LAYER))
				{
					Conv_Layer* last_layer = (Conv_Layer*) layer_list->at(i-1);
					int prev_no_features = last_layer->getNoFeatureMaps();
					int prev_dim_x = last_layer->getXSize();
					int prev_dim_y = last_layer->getYSize();
					int dim_x = (prev_dim_x / pooling_layer->getXReceptive());
					int dim_y = (prev_dim_y / pooling_layer->getYReceptive());

					pooling_layer->setXSize(dim_x);
					pooling_layer->setYSize(dim_y);

					for (int i = 0; i < prev_no_features; i++)
					{
						node_list->push_back(new Matrix(dim_y, dim_x));
					}
				}
				else
				{
					return false;
				}
				break;
			}
			case FULLY_CONNECTED_LAYER:
			{
				node_list->push_back(new Matrix(layer->getSize(), 1));
				weight_list->push_back(new Matrix(layer->getSize(),layer_list->at(i-1)->getSize()));
				bias_list->push_back(new Matrix(layer->getSize(), 1));
				break;
			}
			case DROPOUT_LAYER:
			{
				break;
			}
			default:
			{
				break;
			}
		}
	}
	return true;
}

bool Network::train(...)
{
	return true;
}

float Network::test()
{
	return 0.0f;
}

bool Network::forward(...)
{
	/* Indices to iterate through weight_list, node_list and bias_list */
	int weight_index = 0;
	int bias_index = 0;
	int node_index = 0;

	int no_layers = layer_list->size();

	for(int i = 0; i < no_layers; i++)
	{
		switch (layer_list->at(i)->getLayerType())
		{
			case INPUT_LAYER:
			{
				/* nothing to compute here */
				node_index++;
				break;
			}
			case CONV_LAYER:
			{
				if(layer_list->at(i-1)->getLayerType() == INPUT_LAYER)
				{
					Input_Layer* input_layer = (Input_Layer*) layer_list->at(i-1);
					Conv_Layer*  conv_layer  = (Conv_Layer*) layer_list->at(i);
//					int input_x_size = input_layer->getCols();
//					int input_y_size = input_layer->getRows();
					int x_steps = conv_layer->getXSize();
					int y_steps = conv_layer->getYSize();
					int no_feature_maps = conv_layer->getNoFeatureMaps();
					int x_receptive = conv_layer->getXReceptive();
					int y_receptive = conv_layer->getYReceptive();

					for(int h = 0; h < no_feature_maps; h++)
					{
						for(int j = 0; j < y_steps; j++)
						{
							for(int k = 0; k < x_steps; k++)
							{
								Matrix* node_vector = new Matrix(conv_layer->getXReceptive()*
																		conv_layer->getYReceptive(),1);
								for(int l = 0; l < y_receptive; l++)
								{
									for(int m = 0; m < x_receptive; m++)
									{
										node_vector->set(l*conv_layer->getXReceptive()+m, 1 ,
												node_list->at(node_index-1)->get(j+l, k+m));
									}
								}
								Matrix node_value = (*weight_list->at(weight_index)) * (*node_vector);
								node_list->at(node_index)->set(j, k, node_value.get(0,0));
							}
						}

						(*node_list->at(node_index)) = ( (*(node_list->at(node_index))) + (*(bias_list->at(bias_index))) );
						mathematisches::sigmoid(node_list->at(node_index)->get(), node_list->at(node_index)->get(),
								conv_layer->getXReceptive()*conv_layer->getYReceptive());

						weight_index++;
						bias_index++;
						node_index++;
					}
				}
				else if (layer_list->at(i-1)->getLayerType() == POOLING_LAYER)
				{
					MaxPooling_Layer* input_layer = (MaxPooling_Layer*) layer_list->at(i-1);
					Conv_Layer*  conv_layer  = (Conv_Layer*) layer_list->at(i);
//					int input_x_size = input_layer->getXSize();
//					int input_y_size = input_layer->getYSize();
					int x_steps = conv_layer->getXSize();
					int y_steps = conv_layer->getYSize();
					int no_feature_maps_conv = conv_layer->getNoFeatureMaps();
					int no_feature_maps_pool = input_layer->getNoFeatures();
					int x_receptive = conv_layer->getXReceptive();
					int y_receptive = conv_layer->getYReceptive();
					int prev_node_index = node_index - no_feature_maps_pool; /* node_index is positioned at current node matrix,
						this line calculates the index of the first matrix of the previous pooling layer */

					for(int h = 0; h < no_feature_maps_conv; h++)
					{
						for(int j = 0; j < y_steps; j++)
						{
							for(int k = 0; k < x_steps; k++)
							{
								Matrix* node_vector = new Matrix(x_receptive * y_receptive *
																		no_feature_maps_pool,1);
								for(int n = 0; n < no_feature_maps_pool; n++)
								{
									for(int l = 0; l < y_receptive; l++)
									{
										for(int m = 0; m < x_receptive; m++)
										{
											/* getting node values from different feature maps of previous pooling layer (n) */
											node_vector->set(n*y_receptive*x_receptive + l*x_receptive+m, 1 ,
													node_list->at(prev_node_index + n)->get(j+l, k+m));
										}
									}
								}
								Matrix node_value = (*weight_list->at(weight_index)) * *node_vector;
								node_list->at(node_index)->set(j, k, node_value.get(0,0));
							}
						}

						(*node_list->at(node_index)) = ( (*(node_list->at(node_index))) + (*(bias_list->at(bias_index))) );
						mathematisches::sigmoid(node_list->at(node_index)->get(), node_list->at(node_index)->get(),
								conv_layer->getXReceptive()*conv_layer->getYReceptive());

						weight_index++;
						bias_index++;
						node_index++;
					}
				}
				else
				{
					return false;
				}
				break;
			}
			case POOLING_LAYER:
			{
				if(layer_list->at(i-1)->getLayerType() == CONV_LAYER)
				{
					Conv_Layer* last_layer = (Conv_Layer*) layer_list->at(i-1);
					MaxPooling_Layer* pooling_layer = (MaxPooling_Layer*) layer_list->at(i);
					int conv_x_size = last_layer->getXSize();
					int conv_y_size = last_layer->getYSize();
					int x_step_size = pooling_layer->getXReceptive();
					int y_step_size = pooling_layer->getYReceptive();
					float max_node = 0.0f;
					float new_node = 0.0f;

					for(int j = 0; j < conv_y_size; j=j+y_step_size)
					{
						for(int k = 0; k < conv_x_size; k=k+x_step_size)
						{
							max_node = 0.0f;
							for(int l = 0; l < y_step_size; l++)
							{
								for(int m = 0; m < x_step_size; m++)
								{
									new_node = node_list->at(node_index-1)->get(j+l,k+m);
									if(new_node >= max_node)
									{
										max_node = new_node;
									}
								}
							}
							node_list->at(node_index)->set(j/y_step_size,k/x_step_size, max_node);
						}
					}
					node_index++;
				}
				else
				{
					return false;
				}
				break;
			}
			case FULLY_CONNECTED_LAYER:
			{
				if(layer_list->at(i-1)->getLayerType() == CONV_LAYER)
				{
					Conv_Layer* last_layer = (Conv_Layer*) layer_list->at(i-1);
					Matrix* full_conv_matrix = new Matrix(last_layer->getSize(), 1);

					int no_feature_maps = last_layer->getNoFeatureMaps();
					int y_size = last_layer->getYSize();
					int x_size = last_layer->getXSize();
					int temp_node_index = node_index - no_feature_maps;

					for(int j = 0; j < no_feature_maps; j++)
					{
						for(int k = 0; k < y_size; k++)
						{
							for(int l = 0; l < x_size; l++)
							{
								full_conv_matrix->set(j*y_size*x_size + k * x_size + l, 1,
										node_list->at(temp_node_index)->get(k, l));
							}
						}
						temp_node_index++;
					}

					(*node_list-->at(node_index)) = (*(weight_list->at(weight_index))) * (*full_conv_matrix)
																			+ (*(bias_list->at(bias_index)));

					node_index++;
					bias_index++;
					weight_index++;
				}
				else if (layer_list->at(i-1)->getLayerType() == POOLING_LAYER)
				{
					MaxPooling_Layer* last_layer = (MaxPooling_Layer*) layer_list->at(i-1);
					Matrix* full_pool_matrix = new Matrix(last_layer->getSize(), 1);

					int no_feature_maps = last_layer->getNoFeatures();
					int y_size = last_layer->getYSize();
					int x_size = last_layer->getXSize();
					int temp_node_index = node_index - no_feature_maps;

					for(int j = 0; j < no_feature_maps; j++)
					{
						for(int k = 0; k < y_size; k++)
						{
							for(int l = 0; l < x_size; l++)
							{
								full_pool_matrix->set(j*y_size*x_size + k * x_size + l, 1,
										node_list->at(temp_node_index)->get(k, l));
							}
						}
						temp_node_index++;
					}

					(*node_list->at(node_index)) = (*(weight_list->at(weight_index))) * (*full_pool_matrix)
																			+ (*(bias_list->at(bias_index)));

					node_index++;
					bias_index++;
					weight_index++;
				}
				else if (layer_list->at(i-1)->getLayerType() == FULLY_CONNECTED_LAYER)
				{
					(*node_list->at(node_index)) = (*weight_list->at(weight_index)) * (*node_list->at(node_index-1))
														+ (*bias_list->at(bias_index));
					node_index++;
					bias_index++;
					weight_index++;
				}
				else
				{
					return false;
				}
				break;
			}
			case DROPOUT_LAYER:
			{
				break;
			}
			default:
			{
				return false;
				break;
			}
		}
	}
	return true;
}

bool Network::backpropagate(...)
{
	return true;
}




