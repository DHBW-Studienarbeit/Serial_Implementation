//============================================================================
// Name        : CNN_seriell.cpp
// Author      : Josua Benz, Benjamin Riedle, Florian Schmidt
// Version     :
// Copyright   : Open Source - Take what you can get
// Description : CNN as a serial implementation
//============================================================================

#include <iostream>
#include <string>
#include "./testfile.h"
#include "matrix.hpp"

using namespace std;

int main(int argc, char **argv) {
	string str = getString();
	cout << str << endl; // prints Fuck off eclipse
	Matrix a(3,5);
	a.test();
	a.printOut();
	Matrix b(4,3);
	b.test();
	b.printOut();
	Matrix c = a*b;
	c.printOut();
	a.trans();
	a.printOut();

	/*Matrix a(3,3);
	a.test();
	a.print_out();
	Matrix b(3,3);
	b.test();
	b.print_out();
	a.add(&b);
	a.print_out();

	Matrix a(3,5);
	a.test();
	a.print_out();
	Matrix b(3,5);
	b.test();
	b.print_out();
	Matrix c=a+b;
	c.print_out();
	//c.mul_skalar(3);
	c=3*a;
	c.print_out();*/
	return 0;
}
