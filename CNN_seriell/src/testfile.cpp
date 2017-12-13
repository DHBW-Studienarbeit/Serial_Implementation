/*
 * testfile.cpp
 *
 *  Created on: 23.11.2017
 *      Author: benjamin
 */

#include <string>
#include "PictureContainer.h"
#include "Picture.h"
using namespace std;

string getString() {
	PictureContainer *dataspace = new PictureContainer("./train", 55);
	Picture *first = dataspace->get_nextpicture();
	first = dataspace->get_nextpicture();
	float *input_data = first->get_input();
	return std::to_string(input_data[207]);
}

