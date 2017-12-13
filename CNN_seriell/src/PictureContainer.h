/*
 * PictureContainer.h
 *
 *  Created on: 03.12.2017
 *      Author: Florian
 */

#ifndef PICTURECONTAINER_H_
#define PICTURECONTAINER_H_

#include "Picture.h"

#define PICS_PER_FILE 1000

class PictureContainer {
private:
	Picture images[PICS_PER_FILE] = { Picture() };
	int next_index;
	int file_index;
	int num_of_files;
	std::string foldername;
	void load_pictures();
public:
	PictureContainer(std::string foldername, int num_of_files);
	virtual ~PictureContainer();
	Picture *get_nextpicture(void);
};

#endif /* PICTURECONTAINER_H_ */
