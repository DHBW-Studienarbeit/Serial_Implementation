/*
 * PictureContainer.cpp
 *
 *  Created on: 03.12.2017
 *      Author: Florian
 */

#include "PictureContainer.h"
#include <fstream>



PictureContainer::PictureContainer(std::string foldername, int num_of_files)
{
	this->next_index = 0;
	this->file_index = 0;
	this->foldername = foldername;
	this->num_of_files = num_of_files;
	load_pictures();
}

PictureContainer::~PictureContainer() {

}

void PictureContainer::load_pictures() {
	std::string csv_file = this->foldername + "/" + std::to_string(this->file_index) + ".csv";
	std::ifstream infile(csv_file);
	for(int i=0; i<PICS_PER_FILE; i++)
	{
		std::string line;
		std::getline(infile,line);
		this->images[i] = Picture(&line);
	}
}

Picture * PictureContainer::get_nextpicture(void)
{
	next_index++;
	if(next_index > PICS_PER_FILE)
	{
		next_index=0;
		file_index++;
		if(file_index > num_of_files)
		{
			file_index = 0;
		}
		load_pictures();
	}
	return this->images + next_index;
}
