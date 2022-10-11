#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <bitset>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <typeinfo>

using namespace cv;

#define N 8

int main() {
    // read .tif image
    // To Do: Scan the folder and all sub-folders
    //        to load all data
    std::string image_path = samples::findFile("test.tif");

    // read the image - [256, 256, 3] in this dataset
    // unsigned 8-bit (0~255), 3 channel
    // MRI are grey images, but treat them as 3-channel color image
    // to fit the UNet.
	Mat img = imread(image_path, IMREAD_COLOR);

    // point array pointer to the pointer of image data
    // Note: uint8 is char, print result may be strange
    unsigned char* array = (unsigned char*)img.data;
    uint length = img.total() * img.channels();

    // Conver to binary using C to speed up 
    FILE *fp = fopen("test.dat", "w");
    
    for (int i = 0; i < length; ++i) {
        std::bitset<N> bit_serial(array[i]);
        for (int j = 0; j < N; ++j) {
            frintf(fp, "%d\n", (int)bit_serial[j]);
        }
    }
    fclose(fp);
    
    // free memory
    img.release();

    return 0;
}
