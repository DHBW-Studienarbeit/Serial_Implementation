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
	node_deriv_list = new vector<Matrix*>();
	weight_deriv_list = new vector<Matrix*>();
	bias_deriv_list = new vector<Matrix*>();
	train_picture_container = new PictureContainer("./train", 55);
	test_picture_container = new PictureContainer("./test", 10);
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
		delete node_deriv_list->at(i);
	}

	for(unsigned int i = 0; i < layer_list->size(); i++)
	{
		delete weight_list->at(i);
		delete weight_deriv_list->at(i);
	}

	for(unsigned int i = 0; i < layer_list->size(); i++)
	{
		delete bias_list->at(i);
		delete bias_deriv_list->at(i);
	}

	delete layer_list;
	delete node_list;
	delete weight_list;
	delete bias_list;
	delete node_deriv_list;
	delete weight_deriv_list;
	delete bias_deriv_list;
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
				node_deriv_list->push_back(new Matrix(input_layer->getRows(),input_layer->getCols()));
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

					// added on 2018-02-18
					weight_list->push_back(new Matrix(conv_layer->getNoFeatureMaps(), conv_layer->getXReceptive()*conv_layer->getYReceptive()));
					weight_deriv_list->push_back(new Matrix(conv_layer->getNoFeatureMaps(), conv_layer->getXReceptive()*conv_layer->getYReceptive()));
					bias_list->push_back(new Matrix(conv_layer->getNoFeatureMaps(), 1));
					bias_deriv_list->push_back(new Matrix(conv_layer->getNoFeatureMaps(), 1));

					for (int i = 0; i < conv_layer->getNoFeatureMaps(); i++)
					{
						node_list->push_back(new Matrix(dim_y, dim_x));
						node_deriv_list->push_back(new Matrix(dim_y, dim_x));
						//weight_deriv_list->push_back(new Matrix(conv_layer->getXReceptive(), conv_layer->getYReceptive()));
						//bias_deriv_list->push_back(new Matrix(dim_y, dim_x));
						//weight_list->push_back(new Matrix(conv_layer->getXReceptive(), conv_layer->getYReceptive()));
						//bias_list->push_back(new Matrix(dim_y, dim_x));
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

					// added on 2018-02-18
					int prev_no_feat = last_layer->getNoFeatures();
					weight_list->push_back(new Matrix(conv_layer->getNoFeatureMaps(), conv_layer->getXReceptive()*conv_layer->getYReceptive()*prev_no_feat));
					weight_deriv_list->push_back(new Matrix(conv_layer->getNoFeatureMaps(), conv_layer->getXReceptive()*conv_layer->getYReceptive()*prev_no_feat));
					bias_list->push_back(new Matrix(conv_layer->getNoFeatureMaps(), 1));
					bias_deriv_list->push_back(new Matrix(conv_layer->getNoFeatureMaps(), 1));

					for (int i = 0; i < conv_layer->getNoFeatureMaps(); i++)
					{
						node_list->push_back(new Matrix(dim_y, dim_x));
						node_deriv_list->push_back(new Matrix(dim_y, dim_x));
						//weight_deriv_list->push_back(new Matrix(conv_layer->getXReceptive(), conv_layer->getYReceptive()));
						//bias_deriv_list->push_back(new Matrix(dim_y, dim_x));
						//weight_list->push_back(new Matrix(conv_layer->getXReceptive(), conv_layer->getYReceptive()));
						//bias_list->push_back(new Matrix(dim_y, dim_x));
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
						node_deriv_list->push_back(new Matrix(dim_y, dim_x));
					}

					pooling_layer->setSize(dim_x*dim_y*prev_no_features);
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
				node_deriv_list->push_back(new Matrix(layer->getSize(), 1));
				weight_deriv_list->push_back(new Matrix(layer->getSize(),layer_list->at(i-1)->getSize()));
				bias_deriv_list->push_back(new Matrix(layer->getSize(), 1));
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

bool Network::train(int batch_size, int no_iterations)
{
	int outer_loop = NO_DATA_D/batch_size;
	float cost_sum = 0.0f;
	bool ret_val = false;

	for(int l = 0; l < no_iterations; l++)
	{
		for(int j = 0; j < outer_loop; j++)
		{
			cost_sum = 0.0f;
			reset_backprop_state();
			for(int i = 0; i < batch_size; i++)
			{
				Picture* picture = train_picture_container->get_nextpicture();
				node_list->at(0)->copy_all(picture->get_input());
				cost_sum += forward(picture->get_output());
				ret_val = backpropagate(picture->get_output());
				if(ret_val == false)
				{
					return ret_val; /* end function if failed */
				}
			}
			gradient_descent(cost_sum/batch_size);
		}
	}
	return ret_val;
}

float Network::test()
{
	int correct_index = 0;
	int calculated_index = 0;
	float max_val = 0.0f;
	int correct_detections = 0;
	for(int i = 0; i < NO_TEST_FILES_D; i++)
	{
		for(int j = 0; j < NO_PICS_PER_FILE_D; j++)
		{
			max_val = 0.0f;
			Picture* picture = train_picture_container->get_nextpicture();
			node_list->at(0)->copy_all(picture->get_input());
			forward(picture->get_output());

			for(int k = 0; k < OUTPUT_SIZE; k++)
			{
				if(picture->get_output()[k] == 1.0f)
				{
					correct_index = k;
				}

				if(node_list->at(node_list->size())->get()[k] > max_val)
				{
					calculated_index = k;
					max_val = node_list->at(node_list->size())->get()[k];
				}
			}

			if(correct_index == calculated_index)
			{
				correct_detections++;
			}
		}
	}
	return correct_detections/(NO_PICS_PER_FILE_D*NO_TEST_FILES_D);
}

float Network::forward(float* labels)
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
					Conv_Layer*  conv_layer  = (Conv_Layer*) layer_list->at(i);
					int x_steps = conv_layer->getXSize();
					int y_steps = conv_layer->getYSize();
					int no_feature_maps = conv_layer->getNoFeatureMaps();
					int x_receptive = conv_layer->getXReceptive();
					int y_receptive = conv_layer->getYReceptive();

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
									node_vector->set(l*conv_layer->getXReceptive()+m, 0 ,
											node_list->at(node_index-1)->get(j+l, k+m));
								}
							}

							Matrix node_tmp = (*(weight_list->at(weight_index))) * (*node_vector);
							Matrix node_results = node_tmp + (*(bias_list->at(bias_index)));
							delete node_vector;
							mathematics::sigmoid(node_results.get(), node_results.get(), no_feature_maps);
							for(int h = 0; h < no_feature_maps; h++)
							{
								node_list->at(node_index+h)->set(j, k, node_results.get(h,0));
							}
						}
					}
					weight_index++;
					bias_index++;
					node_index+=no_feature_maps;
				}
				else if (layer_list->at(i-1)->getLayerType() == POOLING_LAYER)
				{
					MaxPooling_Layer* input_layer = (MaxPooling_Layer*) layer_list->at(i-1);
					Conv_Layer*  conv_layer  = (Conv_Layer*) layer_list->at(i);
					int x_steps = conv_layer->getXSize();
					int y_steps = conv_layer->getYSize();
					int no_feature_maps_conv = conv_layer->getNoFeatureMaps();
					int no_feature_maps_pool = input_layer->getNoFeatures();
					int x_receptive = conv_layer->getXReceptive();
					int y_receptive = conv_layer->getYReceptive();
					int prev_node_index = node_index - no_feature_maps_pool; /* node_index is positioned at current node matrix,
						this line calculates the index of the first matrix of the previous pooling layer */

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
										node_vector->set(n*y_receptive*x_receptive + l*x_receptive+m, 0 ,
												node_list->at(prev_node_index + n)->get(j+l, k+m));
									}
								}
							}
							Matrix node_tmp = (*(weight_list->at(weight_index))) * (*node_vector);
							Matrix node_results = node_tmp + (*(bias_list->at(bias_index)));
							delete node_vector;

							mathematics::sigmoid(node_results.get(), node_results.get(), no_feature_maps_conv);

							for(int h = 0; h < no_feature_maps_conv; h++)
							{
								node_list->at(node_index+h)->set(j, k, node_results.get(h,0));
							}
						}
					}

					weight_index++;
					bias_index++;
					node_index+=no_feature_maps_conv;
				}
				else
				{
					return -1.0f;
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
					return -1.0f;
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
								full_conv_matrix->set(j*y_size*x_size + k * x_size + l, 0,
										node_list->at(temp_node_index)->get(k, l));
							}
						}
						temp_node_index++;
					}
					Matrix tmp = (*(weight_list->at(weight_index))) * (*full_conv_matrix);
					(*node_list->at(node_index)) = tmp + (*(bias_list->at(bias_index)));

					delete full_conv_matrix;

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
								full_pool_matrix->set(j*y_size*x_size + k * x_size + l, 0,
										node_list->at(temp_node_index)->get(k, l));
							}
						}
						temp_node_index++;
					}
					Matrix tmp = (*(weight_list->at(weight_index))) * (*full_pool_matrix);
					(*node_list->at(node_index)) = tmp + (*(bias_list->at(bias_index)));

					delete full_pool_matrix;

					node_index++;
					bias_index++;
					weight_index++;
				}
				else if (layer_list->at(i-1)->getLayerType() == FULLY_CONNECTED_LAYER)
				{
					Matrix tmp = (*weight_list->at(weight_index)) * (*node_list->at(node_index-1));
					(*node_list->at(node_index)) = tmp + (*(bias_list->at(bias_index)));
					node_index++;
					bias_index++;
					weight_index++;
				}
				else
				{
					return -1.0f;
				}
				break;
			}
			case DROPOUT_LAYER:
			{
				break;
			}
			default:
			{
				return -1.0f;
				break;
			}
		}
	}
	return mathematics::get_cost(node_list->at(node_list->size()-1)->get(), labels, OUTPUT_SIZE);
}

bool Network::backpropagate(float* labels)
{
	/* Indices to iterate backwards through weight_list, node_list and bias_list */
	// derivation indexes are equal to corresponding indexes
	int weight_index = weight_list->size()-1;
	int bias_index = bias_list->size()-1;
	int node_index = node_list->size()-1;
	// number of layers
	int no_layers = layer_list->size();

	// prepare derivation of last layer's activation
	mathematics::get_cost_derivatives(	node_list->at(node_index)->get(),
										labels,
										node_deriv_list->at(node_index)->get(),
										layer_list->at(no_layers-1)->getSize() );

	// actual backpropagation
	for(int i = no_layers-1; i > 0; i--)
	{
		switch (layer_list->at(i)->getLayerType())
		{
			case INPUT_LAYER:
				//input layer is ignored
				break;
			case POOLING_LAYER:

				if(layer_list->at(i-1)->getLayerType() == CONV_LAYER)
				{
					Conv_Layer* last_layer = (Conv_Layer*) layer_list->at(i-1);
					Matrix* pool_conv_matrix = new Matrix(last_layer->getSize(), 1);
					int no_feature_maps = last_layer->getNoFeatureMaps();
					int y_size = last_layer->getYSize();
					int y_step_size = last_layer->getYReceptive();
					int x_size = last_layer->getXSize();
					int x_step_size = last_layer->getXReceptive();
					int temp_node_index = node_index - no_feature_maps;

					//Über alle Feature Maps gehen um zu ermitteln, welche an jeder Stelle höchsten Wert hat

					for(int j = 0; j < y_size; j=j+y_step_size)
					{
						for(int k = 0; k < x_size; k=k+x_step_size)
						{

							int max_node_index = 0;
							float max_node_value=0.0f;

							for(int l = 0; l < y_step_size; l++)
							{
								for(int m = 0; m < x_step_size; m++)
								{
									for(int n = 0; n < no_feature_maps; n++)
									{
										if(max_node_value < node_list->at(temp_node_index+n)->
												get(j+l,k+m))
										{
											max_node_value = node_list->at(temp_node_index+n)->
													get(j+l,k+m);
											max_node_index = m;
										}
									}
								}
							}
							pool_conv_matrix->set(j*y_size*x_size + k * x_size,0,max_node_index);
						}
					}

					//Wert durchreichen wenn höchster Wert, ansonsten 0
					for(int j = 0; j < y_size; j=j+y_step_size)
					{
						for(int k = 0; k < x_size; k=k+x_step_size)
						{
							for(int m = 0; m < no_feature_maps; m++)
							{
								if(pool_conv_matrix->get(j*y_size*x_size + k * x_size,0) == m){
									node_deriv_list->at(temp_node_index + m)->
											set(j,k,node_deriv_list->at(node_index)->get(j,k));
								}
								else{
									node_deriv_list->at(temp_node_index + m)->set(j,k,0);
								}
							}
						}
					}

				}

				node_index--;

				break;
			case FULLY_CONNECTED_LAYER:

				if(layer_list->at(i-1)->getLayerType() == CONV_LAYER)
				{
					Conv_Layer* last_layer = (Conv_Layer*) layer_list->at(i-1);
					Matrix* full_conv_matrix = new Matrix(last_layer->getSize(), 1);
					Matrix* full_conv_matrix_deriv = new Matrix(last_layer->getSize(), 1);
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
								full_conv_matrix->set(j*y_size*x_size + k * x_size + l, 0,
										node_list->at(temp_node_index)->get(k, l));
							}
						}
						temp_node_index++;
					}

					layer_list->at(i)->backpropagate(full_conv_matrix,
										node_list->at(node_index),
										full_conv_matrix_deriv,
										node_deriv_list->at(node_index),
										weight_list->at(weight_index),
										bias_list->at(bias_index),
										weight_deriv_list->at(weight_index),
										bias_deriv_list->at(bias_index) );

					temp_node_index = node_index - no_feature_maps;
					for(int j = 0; j < no_feature_maps; j++)
					{
						for(int k = 0; k < y_size; k++)
						{
							for(int l = 0; l < x_size; l++)
							{
								node_list->at(temp_node_index)->set(k, l,
										full_conv_matrix_deriv->get(j*y_size*x_size + k * x_size + l, 0));
							}
						}
						temp_node_index++;
					}

					delete full_conv_matrix;
					delete full_conv_matrix_deriv;

					node_index--;
					bias_index--;
					weight_index--;
				}
				else if (layer_list->at(i-1)->getLayerType() == POOLING_LAYER)
				{
					MaxPooling_Layer* last_layer = (MaxPooling_Layer*) layer_list->at(i-1);
					Matrix* full_pool_matrix = new Matrix(last_layer->getSize(), 1);
					Matrix* full_pool_matrix_deriv = new Matrix(last_layer->getSize(), 1);
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
								full_pool_matrix->set(j*y_size*x_size + k * x_size + l, 0,
										node_list->at(temp_node_index)->get(k, l));
							}
						}
						temp_node_index++;
					}

					layer_list->at(i)->backpropagate(full_pool_matrix,
										node_list->at(node_index),
										full_pool_matrix_deriv,
										node_deriv_list->at(node_index),
										weight_list->at(weight_index),
										bias_list->at(bias_index),
										weight_deriv_list->at(weight_index),
										bias_deriv_list->at(bias_index) );

					temp_node_index = node_index - no_feature_maps;
					for(int j = 0; j < no_feature_maps; j++)
					{
						for(int k = 0; k < y_size; k++)
						{
							for(int l = 0; l < x_size; l++)
							{
								node_list->at(temp_node_index)->set(k, l,
										full_pool_matrix_deriv->get(j*y_size*x_size + k * x_size + l, 0));
							}
						}
						temp_node_index++;
					}

					delete full_pool_matrix;
					delete full_pool_matrix_deriv;

					node_index--;
					bias_index--;
					weight_index--;
				}
				else if (layer_list->at(i-1)->getLayerType() == FULLY_CONNECTED_LAYER)
				{
					layer_list->at(i)->backpropagate(node_list->at(node_index-1),
										node_list->at(node_index),
										node_deriv_list->at(node_index-1),
										node_deriv_list->at(node_index),
										weight_list->at(weight_index),
										bias_list->at(bias_index),
										weight_deriv_list->at(weight_index),
										bias_deriv_list->at(bias_index) );
					node_index--;
					bias_index--;
					weight_index--;
				}
				else
				{
					return false;
				}

				break;
			case CONV_LAYER:

				if(layer_list->at(i-1)->getLayerType() == INPUT_LAYER)
				{
					Conv_Layer*  conv_layer  = (Conv_Layer*) layer_list->at(i);
					int x_steps = conv_layer->getXSize();
					int y_steps = conv_layer->getYSize();
					int no_feature_maps = conv_layer->getNoFeatureMaps();
					int x_receptive = conv_layer->getXReceptive();
					int y_receptive = conv_layer->getYReceptive();

					for(int j = 0; j < y_steps; j++)
					{
						for(int k = 0; k < x_steps; k++)
						{
							Matrix* input_vector = new Matrix(x_receptive*y_receptive,1);
							Matrix* input_deriv_vector = new Matrix(x_receptive*y_receptive,1);
							for(int l = 0; l < y_receptive; l++)
							{
								for(int m = 0; m < x_receptive; m++)
								{
									input_vector->set(l*conv_layer->getXReceptive()+m, 0 ,
											node_list->at(node_index-no_feature_maps)->get(j+l, k+m));
								}
							}

							Matrix *activation_vector = new Matrix(no_feature_maps, 1);
							Matrix *activation_deriv_vector = new Matrix(no_feature_maps, 1);
							for(int h = no_feature_maps; h > 0; h--)
							{
								activation_vector->set(h-1, 0, node_list->at(node_index-no_feature_maps+h)->get(j,k));
								activation_deriv_vector->set(h-1, 0, node_deriv_list->at(node_index-no_feature_maps+h)->get(j,k));
							}

							layer_list->at(i)->backpropagate(input_vector,
												activation_vector,
												input_deriv_vector,
												activation_deriv_vector,
												weight_list->at(weight_index),
												bias_list->at(bias_index),
												weight_deriv_list->at(weight_index),
												bias_deriv_list->at(bias_index) );

							delete input_vector;
							delete activation_vector;
							delete activation_deriv_vector;
							// move input_deriv_items to corresponding node_deriv_list matrix member
							for(int l = 0; l < y_receptive; l++)
							{
								for(int m = 0; m < x_receptive; m++)
								{
									node_deriv_list->at(node_index-no_feature_maps)->set(j+l, k+m,
											input_deriv_vector->get(l*conv_layer->getXReceptive()+m, 0));
								}
							}
							delete input_deriv_vector;

						}
					}
					weight_index--;
					bias_index--;
					node_index-=no_feature_maps;
				}
				else if (layer_list->at(i-1)->getLayerType() == POOLING_LAYER)
				{
					MaxPooling_Layer* input_layer = (MaxPooling_Layer*) layer_list->at(i-1);
					Conv_Layer*  conv_layer  = (Conv_Layer*) layer_list->at(i);
					int x_steps = conv_layer->getXSize();
					int y_steps = conv_layer->getYSize();
					int no_feature_maps_conv = conv_layer->getNoFeatureMaps();
					int no_feature_maps_pool = input_layer->getNoFeatures();
					int x_receptive = conv_layer->getXReceptive();
					int y_receptive = conv_layer->getYReceptive();
					int prev_node_index = node_index + 1 - no_feature_maps_conv - no_feature_maps_pool; /* node_index is positioned at current node matrix,
						this line calculates the index of the first matrix of the previous pooling layer */

					for(int j = 0; j < y_steps; j++)
					{
						for(int k = 0; k < x_steps; k++)
						{
							Matrix* input_vector = new Matrix(no_feature_maps_pool*x_receptive*y_receptive,1);
							Matrix* input_deriv_vector = new Matrix(no_feature_maps_pool*x_receptive*y_receptive,1);
							for(int n = 0; n < no_feature_maps_pool; n++)
							{
								for(int l = 0; l < y_receptive; l++)
								{
									for(int m = 0; m < x_receptive; m++)
									{
										input_vector->set(n*y_receptive*x_receptive + l*x_receptive+m, 0 ,
												node_list->at(prev_node_index + n)->get(j+l, k+m));
									}
								}
							}

							Matrix *activation_vector = new Matrix(no_feature_maps_conv, 1);
							Matrix *activation_deriv_vector = new Matrix(no_feature_maps_conv, 1);
							for(int h = no_feature_maps_conv; h > 0; h--)
							{
								activation_vector->set(h-1, 0, node_list->at(node_index-no_feature_maps_conv+h)->get(j,k));
								activation_deriv_vector->set(h-1, 0, node_deriv_list->at(node_index-no_feature_maps_conv+h)->get(j,k));
							}

							layer_list->at(i)->backpropagate(input_vector,
												activation_vector,
												input_deriv_vector,
												activation_deriv_vector,
												weight_list->at(weight_index),
												bias_list->at(bias_index),
												weight_deriv_list->at(weight_index),
												bias_deriv_list->at(bias_index) );

							delete input_vector;
							delete activation_vector;
							delete activation_deriv_vector;
							// move input_deriv_items to corresponding node_deriv_list matrix member
							for(int n = 0; n < no_feature_maps_pool; n++)
							{
								for(int l = 0; l < y_receptive; l++)
								{
									for(int m = 0; m < x_receptive; m++)
									{
										node_deriv_list->at(prev_node_index + n)->set(j+l, k+m,
												input_deriv_vector->get(n*y_receptive*x_receptive + l*x_receptive + m, 0));
									}
								}
							}
							delete input_deriv_vector;

						}
					}
					weight_index--;
					bias_index--;
					node_index-=no_feature_maps_conv;
				}
				else
				{
					return false;
				}
				break;



			case DROPOUT_LAYER:
				//not implemented
				break;
		}
	}
	return true;
}

void Network::gradient_descent(float cost)
{
	//assuming sizes of weight_list and bias_list are equal
	int no_weighted_layers = weight_list->size()-1;
	for(int i =0; i < no_weighted_layers; i++)
	{
		Matrix *weights = weight_list->at(i);
		Matrix *biases = bias_list->at(i);
		Matrix *weights_deriv = weight_deriv_list->at(i);
		Matrix *biases_deriv = bias_deriv_list->at(i);
		for(int m=0; m<weights->getHeight(); m++)
		{
			for(int n=0; n<weights->getLength(); n++)
			{
				weights->set(m, n, weights->get(m,n) - weights_deriv->get(m,n)*LEARNING_RATE);
			}
			biases->set(m, 0, biases->get(m,0) - biases_deriv->get(m,0)*LEARNING_RATE);
		}
	}

}

void Network::reset_backprop_state(void)
{
	int no_layers = weight_deriv_list->size();
	for(int i = 0; i < no_layers; i++)
	{
		weight_deriv_list->at(i)->set_all_equal(0.0);
		bias_deriv_list->at(i)->set_all_equal(0.0);
	}
}
