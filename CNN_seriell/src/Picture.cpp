/*
 * Picture.cpp
 *
 *  Created on: 03.12.2017
 *      Author: Florian
 */

#include "Picture.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>



Picture::Picture()
{

}

Picture::Picture(std::string *line)
{
	std::stringstream          lineStream(*line);
	std::string                cell;

	for(int i=0; i<INPUT_SIZE; i++)
	{
		std::getline(lineStream,cell, ',');
		this->input_data[i] = std::stof(cell);
	}
	for(int i=0; i<OUTPUT_SIZE; i++)
	{
		std::getline(lineStream,cell, ',');
		this->output_data[i] = (float)std::stod(cell);
	}
}

Picture::~Picture() { }

float *Picture::get_input(void)
{
	return this->input_data;
}

float *Picture::get_output(void)
{
	return this->output_data;
}
