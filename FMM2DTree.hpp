#ifndef _FMM2DTree_HPP__
#define _FMM2DTree_HPP__

#include <vector>
#include <Eigen/Dense>
#include <fstream>

#define EIGEN_DONT_PARALLELIZE
#include <iostream>
#include <cmath>
#include <stdlib.h>     /* srand, rand */
#include <time.h>   

#include "domain2D.hpp"
long double epsilon_f	=	1e-6;
long double epsilon	=	1e-10;
using namespace std;

const double PI	=	3.1415926535897932384;

struct pts2D {
	double x,y;
};

struct charge {
	double q,x,y;
};

struct level_box {
	int level,box;
};





class user_Greens_function: public Greens_function {
public:

	// inputs for integrand  chebnodes
	pts2D chebNode_i;
	pts2D chebNode_j;
	int nChebNodes;
	void initialise(pts2D chebNode_i1, pts2D chebNode_j1, int nChebNodes1, double xcenter1, double ycenter1, double L1, int nNodes1) {
		xcenter	=	xcenter1;
		ycenter	=	ycenter1;
		Lx	=	L1;
		Ly	=	L1;
		nNodes	=	nNodes1;

		chebNode_i	=	chebNode_i1;
		chebNode_j	=	chebNode_j1;
		nChebNodes	=	nChebNodes1;

	}
	
	#ifdef ONEOVERR
	user_Greens_function() {
		isTrans		=	true;
		isHomog		=	true;
		isLogHomog	=	false;
		alpha		=	-1.0;
	};

	

	long double integrand(long double x, long double y) {
		long double R = (sqrt(pow(chebNode_i.x-x,2)+pow(chebNode_i.y-y,2)));
		if (R < 1e-10) {
			return 0.0;
		}
		else {
			return (1/R)  * get_S(chebNode_j.x, x, nChebNodes) * get_S(chebNode_j.y, y, nChebNodes);
		}
	};
	#elif LOGR
	user_Greens_function() {
		isTrans		=	true;
		isHomog		=	false;
		isLogHomog	=	true;
		alpha		=	1.0;
	};
	long double integrand(long double x, long double y) {
		long double R = (chebNode_i.x-x)*(chebNode_i.x-x)+(chebNode_i.y-y)*(chebNode_i.y-y);
		return (0.5*log(R) * get_S(chebNode_j.x, (x-xcenter)/Lx, nChebNodes) * get_S(chebNode_j.y, (y-ycenter)/Ly, nChebNodes));
	};
	#endif

	//	get_ChebPoly
	double get_ChebPoly(double x, int n) {
		return cos(n*acos(x));
	}

	//	get_S
	double get_S(double x, double y, int n) {
		double S	=	0.5;
		for (int k=1; k<n; ++k) {
			S+=get_ChebPoly(x,k)*get_ChebPoly(y,k);
		}
		return 2.0/n*S;
	}

	// RHS function
	double f(double x, double y) {
		double sigma = 0.5;
		return exp(-(x*x+y*y)/(2*sigma*sigma))/(2*PI*sigma*sigma);
		//return (1.0); //constant function
	}

	~user_Greens_function() {};

};




class user_function2: public Greens_function {
public:

	// inputs for integrand  chebnodes
	pts2D chebNode_j;
	int nChebNodes;
	

	user_function2() {
		xcenter	=	0.0;
		ycenter	=	0.0;
		Lx	=	1.0;
		Ly	=	1.0;
		
	};
	void initialise (pts2D chebNode_j1, int nChebNodes1, int nNodes1) {
		nNodes	=	nNodes1;

		chebNode_j	=	chebNode_j1;
		nChebNodes	=	nChebNodes1;
	}
	long double integrand(long double x, long double y) {
		return (long double)(get_S(chebNode_j.x, x, nChebNodes) * get_S(chebNode_j.y, y, nChebNodes));
	}


	//	get_ChebPoly
	double get_ChebPoly(double x, int n) {
		return cos(n*acos(x));
	}

	//	get_S
	double get_S(double x, double y, int n) {
		double S	=	0.5;
		for (int k=1; k<n; ++k) {
			S+=get_ChebPoly(x,k)*get_ChebPoly(y,k);
		}
		return 2.0/n*S;
	}

	~user_function2() {};

};



class user_function3: public Greens_function {
public:

	// inputs for integrand  chebnodes
	pts2D chebNode_i;
	int nChebNodes;
	

	user_function3() {
		
	};
	
	void initialise(pts2D chebNode_i1, long double xcenter1, long double ycenter1, long double L1, int nNodes1) {
		xcenter	=	xcenter1;
		ycenter	=	ycenter1;
		Lx	=	L1;
		Ly	=	L1;
		nNodes	=	nNodes1;
		chebNode_i	=	chebNode_i1;
	}

	long double integrand(long double x, long double y) {
		long double R = pow(chebNode_i.x-x,2)+pow(chebNode_i.y-y,2);
		return 0.5*log(R) * f(x,y) ;
	}

	// RHS function
	double f(double x, double y) {
		double sigma = 0.5;
		return exp(-(x*x+y*y)/(2*sigma*sigma))/(2*PI*sigma*sigma);
		//return (1); //constant function
	}
	~user_function3() {};

};




class FMM2DBox {
public:
	int boxNumber;
	int parentNumber;
	int childrenNumbers[4];
	int neighborNumbers[8];
	int innerNumbers[16];
	int outerNumbers[24];

	
	FMM2DBox () {
		boxNumber		=	-1;
		parentNumber	=	-1;
		for (int l=0; l<4; ++l) {
			childrenNumbers[l]	=	-1;
		}
		for (int l=0; l<8; ++l) {
			neighborNumbers[l]	=	-1;
		}
		for (int l=0; l<16; ++l) {
			innerNumbers[l]		=	-1;
		}
		for (int l=0; l<24; ++l) {
			outerNumbers[l]		=	-1;
		}
	}
	
	Eigen::VectorXd multipoles;
	Eigen::VectorXd locals;

	pts2D center;

	//	The following will be stored only at the leaf nodes
	std::vector<pts2D> chebNodes;
};



template <typename Greens_functiontype>
class FMM2DTree {
public:
	Greens_functiontype* K;
	int nLevels;			//	Number of levels in the tree.
	int nChebNodes;			//	Number of Chebyshev nodes along one direction.
	int rank;				//	Rank of interaction, i.e., rank = nChebNodes*nChebNodes.
	int N;					//	Number of particles.
	double L;				//	Semi-length of the simulation box.
	double smallestBoxSize;	//	This is L/2.0^(nLevels).
	double a;				//	Cut-off for self-interaction. This is less than the length of the smallest box size.

	std::vector<int> nBoxesPerLevel;			//	Number of boxes at each level in the tree.
	std::vector<double> boxRadius;				//	Box radius at each level in the tree assuming the box at the root is [-1,1]^2
	std::vector<double> boxHomogRadius;			//	Stores the value of boxRadius^{alpha}
	std::vector<double> boxLogHomogRadius;		//	Stores the value of alpha*log(boxRadius)
	std::vector<std::vector<FMM2DBox> > tree;	//	The tree storing all the information.
	
	//	childless_boxes
	std::vector<level_box> childless_boxes;

	//	charge in coloumbs and its location
//	std::vector<charge> charge_database;
  	
	//	Chebyshev nodes
	std::vector<double> standardChebNodes1D;
	std::vector<pts2D> standardChebNodes;
	std::vector<pts2D> standardChebNodesChild;
	std::vector<pts2D> leafChebNodes;

	//	Different Operators
	Eigen::MatrixXd selfInteraction;		//	Needed only at the leaf level.
	Eigen::MatrixXd neighborInteraction[8];	//	Neighbor interaction only needed at the leaf level.
	Eigen::MatrixXd M2M[4];					//	Transfer from multipoles of 4 children to multipoles of parent.
	Eigen::MatrixXd L2L[4];					//	Transfer from locals of parent to locals of 4 children.
	Eigen::MatrixXd M2LInner[16];			//	M2L of inner interactions. This is done on the box [-L,L]^2.
	Eigen::MatrixXd M2LOuter[24];			//	M2L of outer interactions. This is done on the box [-L,L]^2.
	Eigen::RowVectorXd M2L_term2;			// same for al M2Ls
	Eigen::MatrixXd S_mat;

	//	the potential evaluated by interpolating the potential at the locals and adding the near field
	Eigen::VectorXd potential;

// public:
	FMM2DTree(Greens_functiontype* K, /*int nLevels,*/ int nChebNodes, double L, int nLevels) {
		this->K						=	K;
		//this->nLevels			=	nLevels;
		this->nChebNodes			=	nChebNodes;
		this->rank					=	nChebNodes*nChebNodes;
		this->L						=	L;
		this->nLevels					=	nLevels;
		nBoxesPerLevel.push_back(1);
		boxRadius.push_back(L);
		boxHomogRadius.push_back(pow(L,K->alpha));
		boxLogHomogRadius.push_back(K->alpha*log(L));
		for (int k=1; k<=nLevels; ++k) {
			nBoxesPerLevel.push_back(4*nBoxesPerLevel[k-1]);
			boxRadius.push_back(0.5*boxRadius[k-1]);
			boxHomogRadius.push_back(pow(0.5,K->alpha)*boxHomogRadius[k-1]);
			boxLogHomogRadius.push_back(boxLogHomogRadius[k-1]-K->alpha*log(2));
		}
		this->N					=	rank*nBoxesPerLevel[nLevels];
	}

	std::vector<pts2D> shift_Cheb_Nodes(double xShift, double yShift) {
		std::vector<pts2D> shiftedChebNodes;
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	standardChebNodes[k].x+2*xShift;
			temp.y	=	standardChebNodes[k].y+2*yShift;
			shiftedChebNodes.push_back(temp);
		}
		return shiftedChebNodes;
	}
	
	std::vector<pts2D> shift_Leaf_Cheb_Nodes(double xShift, double yShift) {
		std::vector<pts2D> shiftedChebNodes;
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	leafChebNodes[k].x+xShift;
			temp.y	=	leafChebNodes[k].y+yShift;
			shiftedChebNodes.push_back(temp);
		}
		return shiftedChebNodes;
	}


	//	shifted_scaled_cheb_nodes	//	used in evaluating multipoles
	std::vector<pts2D> shift_scale_Cheb_Nodes(double xShift, double yShift, double radius) {
		std::vector<pts2D> shifted_scaled_ChebNodes;
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	radius*standardChebNodes[k].x+xShift;
			temp.y	=	radius*standardChebNodes[k].y+yShift;
			shifted_scaled_ChebNodes.push_back(temp);
		}
		return shifted_scaled_ChebNodes;
	}


	//	get_ChebPoly
	double get_ChebPoly(double x, int n) {
		return cos(n*acos(x));
	}

	//	get_S
	double get_S(double x, double y, int n) {
		double S	=	0.5;
		for (int k=1; k<n; ++k) {
			S+=get_ChebPoly(x,k)*get_ChebPoly(y,k);
		}
		return 2.0/n*S;
	}
	//	set_Standard_Cheb_Nodes
	void set_Standard_Cheb_Nodes() {
		for (int k=0; k<nChebNodes; ++k) {
			standardChebNodes1D.push_back(-cos((k+0.5)/nChebNodes*PI));
		}
		pts2D temp1;
		for (int j=0; j<nChebNodes; ++j) {
			for (int k=0; k<nChebNodes; ++k) {
				temp1.x	=	standardChebNodes1D[k];
				temp1.y	=	standardChebNodes1D[j];
				standardChebNodes.push_back(temp1);
			}
		}
		//	Left Bottom child, i.e., Child 0
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x-0.5;
				temp1.y	=	0.5*temp1.y-0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Right Bottom child, i.e., Child 1
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x+0.5;
				temp1.y	=	0.5*temp1.y-0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Right Top child, i.e., Child 2
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x+0.5;
				temp1.y	=	0.5*temp1.y+0.5;
				standardChebNodesChild.push_back(temp1);
		}
		//	Left Top child, i.e., Child 3
		for (int j=0; j<rank; ++j) {
				temp1	=	standardChebNodes[j];
				temp1.x	=	0.5*temp1.x-0.5;
				temp1.y	=	0.5*temp1.y+0.5;
				standardChebNodesChild.push_back(temp1);
		}
	}

	void get_Transfer_Matrix() {
		for (int l=0; l<4; ++l) {
			L2L[l]	=	Eigen::MatrixXd(rank,rank);
			for (int j=0; j<rank; ++j) {
				for (int k=0; k<rank; ++k) {
					L2L[l](j,k)	=	get_S(standardChebNodes[k].x, standardChebNodesChild[j+l*rank].x, nChebNodes)*get_S(standardChebNodes[k].y, standardChebNodesChild[j+l*rank].y, nChebNodes);
				}
			}
		}
		for (int l=0; l<4; ++l) {
			M2M[l]	=	L2L[l].transpose();
		}
		/*M2M	=	Eigen::MatrixXd(rank,4*rank);
		for (int j=0; j<rank; ++j) {
			for (int k=0; k<4*rank; ++k) {
				M2M(j,k)	=	get_S(standardChebNodes[j].x, standardChebNodesChild[k].x, nChebNodes)*get_S(standardChebNodes[j].y, standardChebNodesChild[k].y, nChebNodes);
			}
		}*/
	}



	void createTree() {
		//	First create root and add to tree
		FMM2DBox root;
		root.boxNumber		=	0;
		root.parentNumber	=	-1;
		#pragma omp parallel for
		for (int l=0; l<4; ++l) {
			root.childrenNumbers[l]	=	l;
		}
		#pragma omp parallel for
		for (int l=0; l<8; ++l) {
			root.neighborNumbers[l]	=	-1;
		}
		#pragma omp parallel for
		for (int l=0; l<16; ++l) {
			root.innerNumbers[l]	=	-1;
		}
		#pragma omp parallel for
		for (int l=0; l<24; ++l) {
			root.outerNumbers[l]	=	-1;
		}
		std::vector<FMM2DBox> rootLevel;
		rootLevel.push_back(root);
		tree.push_back(rootLevel);
		for (int j=1; j<=nLevels; ++j) {
			std::vector<FMM2DBox> level;
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				FMM2DBox box;
				box.boxNumber		=	k;
				box.parentNumber	=	k/4;
				for (int l=0; l<4; ++l) {
					box.childrenNumbers[l]	=	4*k+l;
				}
				level.push_back(box);
			}
			tree.push_back(level);
		}
	}


	void assign_Center_Location() {
		int J, K;
		tree[0][0].center.x	=	0.0;
		tree[0][0].center.y	=	0.0;
		for (int j=0; j<nLevels; ++j) {
			J	=	j+1;
			double shift	=	0.5*boxRadius[j];
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				K	=	4*k;
				tree[J][K].center.x	=	tree[j][k].center.x-shift;
				tree[J][K+1].center.x	=	tree[j][k].center.x+shift;
				tree[J][K+2].center.x	=	tree[j][k].center.x+shift;
				tree[J][K+3].center.x	=	tree[j][k].center.x-shift;

				tree[J][K].center.y	=	tree[j][k].center.y-shift;
				tree[J][K+1].center.y	=	tree[j][k].center.y-shift;
				tree[J][K+2].center.y	=	tree[j][k].center.y+shift;
				tree[J][K+3].center.y	=	tree[j][k].center.y+shift;
			}
		}
		for (int k=0; k<rank; ++k) {
			pts2D temp;
			temp.x	=	boxRadius[nLevels]*standardChebNodes[k].x;
			temp.y	=	boxRadius[nLevels]*standardChebNodes[k].y;
			leafChebNodes.push_back(temp);
		}
		#pragma omp parallel for
		for (int k=0; k<nBoxesPerLevel[nLevels]; ++k) {
			tree[nLevels][k].chebNodes	=	shift_Leaf_Cheb_Nodes(tree[nLevels][k].center.x, tree[nLevels][k].center.y);
		}
	}


	//	Assigns the interactions for child0 of a box
	void assign_Child0_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k;
		int nN;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N5  |  N4  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  **  |  N3  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[3]	=	nC+1;
			tree[nL][nC].neighborNumbers[4]	=	nC+2;
			tree[nL][nC].neighborNumbers[5]	=	nC+3;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/****************************/
			/*				   ______	*/
			/*				  |		 |	*/
			/*				  |	 **  |	*/
			/*	 _____________|______|  */
			/*	|	   |	  |			*/
			/*	|  I15 |  N0  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I0  |  I1  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[0];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 ______			  		*/
			/*	|	   |	  			*/
			/*	|  **  |				*/
			/*	|______|______			*/
			/*	|	   |	  |			*/
			/*	|  N1  |  N2  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I2  |  I3  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[1]	=	tree[j][nN].childrenNumbers[3];
			}
		}

		//	Assign children of parent's second neighbor
		{
			/************************************/
			/*	 ______			  				*/
			/*	|	   |	  					*/
			/*	|  **  |						*/
			/*	|______|	   _____________	*/
			/*				  |	     |	    |	*/
			/*				  |  I5  |  O8  |	*/
			/*				  |______|______|	*/
			/*				  |	     |	    |	*/
			/*				  |  I4  |  O7  |	*/
			/*				  |______|______|	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[2];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[7]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[8]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/************************************/
			/*				   _____________	*/
			/*				  |	     |	    |	*/
			/*				  |  I7  |  O10 |	*/
			/*	 ______		  |______|______|	*/
			/*	|	   |	  |	     |	    |	*/
			/*	|  **  |	  |  I6  |  O9  |	*/
			/*	|______|	  |______|______|	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[3];
			if (nN!=-1) {
				tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[9]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[10]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[3];
			}

		}

		//	Assign children of parent's fourth neighbor
		{
			/************************************/
			/*				   _____________	*/
			/*				  |	     |	    |	*/
			/*				  |  O13 |  O12 |	*/
			/*				  |______|______|	*/
			/*				  |	     |	    |	*/
			/*		    	  |  I8  |  O11 |	*/
			/*		    	  |______|______|	*/
			/*									*/
			/*									*/
			/*	 ______							*/
			/*  |      |						*/
			/*  |  **  |						*/
			/*  |______|						*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[4];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[11]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[12]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[13]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  O15 |  O14 |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  I10 |  I9  |		*/
			/*	|______|______|		*/
			/*						*/
			/*						*/
			/*	 ______				*/
			/*  |	   |			*/
			/*	|  **  |			*/
			/*	|______|			*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[14]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[15]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/****************************/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  O17 |  O16 |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I12 |  I11 |			*/
			/*	|______|______|			*/
			/*							*/
			/*							*/
			/*				   ______	*/
			/*  			  |		 |	*/
			/*				  |	 **  |	*/
			/*				  |______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[6];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[16]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[17]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/****************************/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I13 |  N6  |			*/
			/*	|______|______|______	*/
			/*  |	   |	  |		 |	*/
			/*	|  I14 |  N7  |	 **  |	*/
			/*	|______|______|______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[7];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[7]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[3];				
			}
		}
	}

	//	Assigns the interactions for child1 of a box
	void assign_Child1_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+1;
		int nN;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N6  |  N5  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N7  |  **  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[7]	=	nC-1;
			tree[nL][nC].neighborNumbers[5]	=	nC+1;
			tree[nL][nC].neighborNumbers[6]	=	nC+2;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/************************************/
			/*				   		  ______	*/
			/*				  	     |		|	*/
			/*				         |	**  |	*/
			/*	 _____________       |______|  	*/
			/*	|	   |	  |					*/
			/*	|  O22 |  I15 |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O23 |  I0  |					*/
			/*	|______|______|					*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[0];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[23]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[22]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 		______		  	*/
			/*		   |	  |			*/
			/*	       |  **  |			*/
			/*	 ______|______|			*/
			/*	|	   |	  |			*/
			/*	|  N0  |  N1  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I1  |  I2  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[1]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's second neighbor
		{
			/****************************/
			/*	 ______		  			*/
			/*	|	   |				*/
			/*	|  **  |	  			*/
			/*	|______|_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  N2  |  I5  |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*	       |  I3  |  I4  |	*/
			/*		   |______|______|	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[2];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  N4  |	 I7	 |  */
			/*	 ______|______|______|	*/
			/*	|	   |	  |	     |	*/
			/*	|  **  |  N3  |  I6  |  */
			/*	|______|______|______| 	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[3];
			if (nN != -1){
				tree[nL][nC].neighborNumbers[3]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fourth neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  O14 |  O13 |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  I9  |  I8  |	*/
			/*		   |______|______|	*/
			/*				  			*/
			/*				  			*/
			/*	 ______					*/
			/*	|	   |				*/
			/*  |  **  |				*/
			/*  |______|				*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[4];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[13]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[14]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  O16 |  O15 |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  I11 |  I10 |		*/
			/*	|______|______|		*/
			/*						*/
			/*						*/
			/*		    ______		*/
			/* 		   |	  |		*/
			/*		   |  **  |		*/
			/*		   |______|		*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[15]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[16]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/************************************/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O18 |  O17 |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O19 |  I12 |					*/
			/*	|______|______|					*/
			/*									*/
			/*									*/
			/*				   		  ______	*/
			/*  			  		 |		|	*/
			/*				  		 |	** 	|	*/
			/*				  		 |______|	*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[6];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[19]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[17]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[18]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/************************************/
			/*									*/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O20 |  I13 |					*/
			/*	|______|______|		  ______	*/
			/*  |	   |	  |		 |		|	*/
			/*	|  O21 |  I14 |	 	 |	**  |	*/
			/*	|______|______|		 |______|	*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[7];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[21]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[20]	=	tree[j][nN].childrenNumbers[3];				
			}
		}
	}

	//	Assigns the interactions for child2 of a box
	void assign_Child2_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+2;
		int nN;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  N7  |  **  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N0  |  N1  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[0]	=	nC-2;
			tree[nL][nC].neighborNumbers[1]	=	nC-1;
			tree[nL][nC].neighborNumbers[7]	=	nC+1;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/************************************/
			/*				   		  ______	*/
			/*				  	     |		|	*/
			/*				         |	**  |	*/
			/*				         |______|  	*/
			/*									*/
			/*									*/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O23 |  I0  |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O0  |  O1  |					*/
			/*	|______|______|					*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[0];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[0]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[1]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[23]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 		______		  	*/
			/*		   |	  |			*/
			/*	       |  **  |			*/
			/*	 	   |______|			*/
			/*							*/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I1  |  I2  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O2  |  O3  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[2]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[3]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's second neighbor
		{
			/****************************/
			/*	 ______		  			*/
			/*	|	   |				*/
			/*	|  **  |	  			*/
			/*	|______|				*/
			/*							*/
			/*							*/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  I3  |  I4  |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*	       |  O4  |  O5  |	*/
			/*		   |______|______|	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[2];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[4]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[5]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/****************************/
			/*	 ____________________	*/
			/*	|	   |	  |	     |	*/
			/*	|  **  |  N3  |	 I6	 |  */
			/*	|______|______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  N2  |  I5  |  */
			/*		   |______|______| 	*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[3];
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[2]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].neighborNumbers[3]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fourth neighbor
		{
			/****************************/
			/*			_____________	*/
			/*		   |	  |	     |	*/
			/*		   |  I9  |  I8  |	*/
			/*		   |______|______|	*/
			/*		   |	  |	     |	*/
			/*		   |  N4  |  I7  |	*/
			/*	 ______|______|______|	*/
			/*	|	   |	  			*/
			/*	|  **  |	  			*/
			/*	|______|	  			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[4];
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  I11 |  I10 |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N6  |  N5  |		*/
			/*	|______|______|		*/
			/*		   |	  |		*/
			/*		   |  **  |		*/
			/*		   |______|		*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[5]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/************************************/
			/*	 _____________					*/
			/*	|	   |	  |					*/
			/*	|  O19 |  I12 |					*/
			/*	|______|______|					*/
			/*	|	   |	  |					*/
			/*	|  O20 |  I13 |					*/
			/*	|______|______|		  ______	*/
			/*  			  		 |		|	*/
			/*				  		 |	** 	|	*/
			/*				  		 |______|	*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[6];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[20]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[19]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/************************************/
			/*									*/
			/*	 _____________		  ______	*/
			/*	|	   |	  |		 |	    |	*/
			/*	|  O21 |  I14 |		 |	**	|	*/
			/*	|______|______|		 |______|	*/
			/*  |	   |	  |		 			*/
			/*	|  O22 |  I15 |	 	 			*/
			/*	|______|______|		 			*/
			/*  								*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[7];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[22]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].outerNumbers[21]	=	tree[j][nN].childrenNumbers[3];				
			}
		}
	}

	//	Assigns the interactions for child3 of a box
	void assign_Child3_Interaction(int j, int k) {
		int nL	=	j+1;
		int nC	=	4*k+3;
		int nN;

		//	Assign siblings
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  **  |  N3  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N1  |  N2  |		*/
			/*	|______|______|		*/
			/*						*/
			/************************/
			tree[nL][nC].neighborNumbers[1]	=	nC-3;
			tree[nL][nC].neighborNumbers[2]	=	nC-2;
			tree[nL][nC].neighborNumbers[3]	=	nC-1;
		}

		//	Assign children of parent's zeroth neighbor
		{
			/****************************/
			/*				   ______	*/
			/*				  |		 |	*/
			/*				  |	 **  |	*/
			/*				  |______|  */
			/*							*/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I0  |  I1  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O1  |  O2  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[0];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[1]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[2]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[1]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[0]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's first neighbor
		{
			/****************************/
			/*	 ______		  			*/
			/*	|	   |				*/
			/*	|  **  |				*/
			/*	|______|				*/
			/*							*/
			/*							*/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I2  |  I3  |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  O3  |  O4  |			*/
			/*	|______|______|			*/
			/*							*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[1];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[3]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[4]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[3]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[2]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's second neighbor
		{
			/************************************/
			/*	 ______		  					*/
			/*	|	   |						*/
			/*	|  **  |	  					*/
			/*	|______|						*/
			/*									*/
			/*									*/
			/*				   _____________	*/
			/*		   		  |	     |	    |	*/
			/*		   		  |  I4  |  O7  |	*/
			/*		   		  |______|______|	*/
			/*		   		  |	  	 |	    |	*/
			/*	       		  |  O5  |  O6  |	*/
			/*		   		  |______|______|	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[2];
			if (nN != -1) {
				tree[nL][nC].outerNumbers[5]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[6]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[7]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[4]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's third neighbor
		{
			/************************************/
			/*	 ______		   _____________	*/
			/*	|	   |	  |	     |		|	*/
			/*	|  **  |      |	 I6	 |  O9	|	*/
			/*	|______|	  |______|______|	*/
			/*		   		  |	  	 |	    |	*/
			/*		   		  |  I5  |  O8  |  	*/
			/*		   		  |______|______| 	*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[3];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[5]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[8]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[9]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[6]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fourth neighbor
		{
			/************************************/
			/*				   _____________	*/
			/*		   		  |	  	 |	    |	*/
			/*		   		  |  I8  |  O11 |	*/
			/*		   		  |______|______|	*/
			/*		   		  |	  	 |	    |	*/
			/*		   		  |  I7  |  O10 |	*/
			/*	 ______	      |______|______|	*/
			/*	|	   |	  					*/
			/*	|  **  |	  					*/
			/*	|______|	  					*/
			/*									*/
			/************************************/
			nN	=	tree[j][k].neighborNumbers[4];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[7]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].outerNumbers[10]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].outerNumbers[11]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[8]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's fifth neighbor
		{
			/************************/
			/*	 _____________		*/
			/*	|	   |	  |		*/
			/*	|  I10 |  I9  |		*/
			/*	|______|______|		*/
			/*	|	   |	  |		*/
			/*	|  N5  |  N4  |		*/
			/*	|______|______|		*/
			/*	|	   |			*/
			/*	|  **  |			*/
			/*	|______|			*/
			/*  					*/
			/************************/
			nN	=	tree[j][k].neighborNumbers[5];
			if (nN != -1) {
				tree[nL][nC].neighborNumbers[5]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[4]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[9]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[10]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's sixth neighbor
		{
			/****************************/
			/*	 _____________			*/
			/*	|	   |	  |			*/
			/*	|  I12 |  I11 |			*/
			/*	|______|______|			*/
			/*	|	   |	  |			*/
			/*	|  I13 |  N6  |			*/
			/*	|______|______|______	*/
			/*  			  |		 |	*/
			/*				  |	 **  |	*/
			/*				  |______|	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[6];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[13]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[6]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].innerNumbers[11]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[12]	=	tree[j][nN].childrenNumbers[3];				
			}
		}

		//	Assign children of parent's seventh neighbor
		{
			/****************************/
			/*							*/
			/*	 ____________________	*/
			/*	|	   |	  |		 |	*/
			/*	|  I14 |  N7  |	 **	 |	*/
			/*	|______|______|______|	*/
			/*  |	   |	  |		 	*/
			/*	|  I15 |  N0  |	 	 	*/
			/*	|______|______|		 	*/
			/*  						*/
			/****************************/
			nN	=	tree[j][k].neighborNumbers[7];
			if (nN != -1) {
				tree[nL][nC].innerNumbers[15]	=	tree[j][nN].childrenNumbers[0];
				tree[nL][nC].neighborNumbers[0]	=	tree[j][nN].childrenNumbers[1];
				tree[nL][nC].neighborNumbers[7]	=	tree[j][nN].childrenNumbers[2];
				tree[nL][nC].innerNumbers[14]	=	tree[j][nN].childrenNumbers[3];				
			}
		}
	}

	//	Assigns the interactions for the children of a box
	void assign_Box_Interactions(int j, int k) {
		assign_Child0_Interaction(j,k);
		assign_Child1_Interaction(j,k);
		assign_Child2_Interaction(j,k);
		assign_Child3_Interaction(j,k);
	}

	//	Assigns the interactions for the children all boxes at a given level
	void assign_Level_Interactions(int j) {
		#pragma omp parallel for
		for (int k=0; k<nBoxesPerLevel[j]; ++k) {
			assign_Box_Interactions(j,k);
		}
	}

	//	Assigns the interactions for the children all boxes in the tree
	void assign_Tree_Interactions() {
		for (int j=0; j<nLevels; ++j) {
			assign_Level_Interactions(j);
		}
	}




	//	Obtain the desired matrix for M2L (uniform GQ)
	void obtain_Desired_Operator(double xcenter_j, double ycenter_j, double L_j, Eigen::MatrixXd& T) {
		T	=	Eigen::MatrixXd(rank,rank);
		//double L_j = 1.0;
		int nNodes		=	6;	
		for (int i=0; i<rank; ++i) {
			for (int j=0; j<rank; ++j) {
				user_Greens_function* mine	=	new user_Greens_function();
				mine->initialise(standardChebNodes[i], standardChebNodes[j], nChebNodes, xcenter_j, ycenter_j, L_j, nNodes);
				mine->obtain_Quadrature_Uniform();
				T(i,j)	=	double(mine->integral);
			}
		}
	}


	//	Obtain the desired matrix for neighbors and self(AGQ)
	void obtain_Desired_Operator_Leaf(double xcenter_j, double ycenter_j, double L_j, Eigen::MatrixXd& T) {
		T	=	Eigen::MatrixXd(rank,rank);
		int nNodes		=	6;	
		for (int i=0; i<rank; ++i) {
			for (int j=0; j<rank; ++j) {
				user_Greens_function* mine	=	new user_Greens_function();
				mine->initialise(standardChebNodes[i], standardChebNodes[j], nChebNodes, xcenter_j, ycenter_j, L_j, nNodes);
				mine->obtain_Quadrature(epsilon);
				T(i,j)	=	double(mine->integral);
			}
		}
	}


	void obtain_Desired_Operator2(Eigen::RowVectorXd& T) {
		T	=	Eigen::RowVectorXd(rank);
		int nNodes		=	6;	
		for (int j=0; j<rank; ++j) {
			user_function2* mine	=	new user_function2();
			mine->initialise(standardChebNodes[j], nChebNodes, nNodes);
			mine->obtain_Quadrature_Uniform();
			T(j)	=	double(mine->integral);
		}
	}


	//	Assemble FMM Operators
	void assemble_Operators_FMM() {
		//	Assemble Outer Interactions
		for (int l=0; l<6; ++l) {
			obtain_Desired_Operator(2.0*(l-3), -6.0, 1.0, M2LOuter[l]);
		}
		for (int l=0; l<6; ++l) {
			obtain_Desired_Operator(6.0, 2.0*(l-3), 1.0, M2LOuter[l+6]);
		}
		for (int l=0; l<6; ++l) {
			obtain_Desired_Operator(2.0*(3-l), 6.0, 1.0, M2LOuter[l+12]);
		}
		for (int l=0; l<6; ++l) {
			obtain_Desired_Operator(-6.0, 2.0*(3-l), 1.0, M2LOuter[l+18]);
		}
		//	Assemble Inner Interactions
		for (int l=0; l<4; ++l) {
			obtain_Desired_Operator(2.0*(l-2), -4.0, 1.0, M2LInner[l]);
		}
		for (int l=0; l<4; ++l) {
			obtain_Desired_Operator(4.0, 2.0*(l-2), 1.0, M2LInner[l+4]);
		}
		for (int l=0; l<4; ++l) {
			obtain_Desired_Operator(2.0*(2-l), 4.0, 1.0, M2LInner[l+8]);
		}
		for (int l=0; l<4; ++l) {
			obtain_Desired_Operator(-4.0, 2.0*(2-l), 1.0, M2LInner[l+12]);
		}
		obtain_Desired_Operator2(M2L_term2);
								
		//	Assemble Neighbor Interactions
		obtain_Desired_Operator_Leaf(-2.0, -2.0, 1.0, neighborInteraction[0]);
		obtain_Desired_Operator_Leaf(0.0, -2.0, 1.0, neighborInteraction[1]);
		obtain_Desired_Operator_Leaf(2.0, -2.0, 1.0, neighborInteraction[2]);
		obtain_Desired_Operator_Leaf(2.0, 0.0, 1.0, neighborInteraction[3]);
		obtain_Desired_Operator_Leaf(2.0, 2.0, 1.0, neighborInteraction[4]);
		obtain_Desired_Operator_Leaf(0.0, 2.0, 1.0, neighborInteraction[5]);
		obtain_Desired_Operator_Leaf(-2.0, 2.0, 1.0, neighborInteraction[6]);
		obtain_Desired_Operator_Leaf(-2.0, 0.0, 1.0, neighborInteraction[7]);
		
		//	Assemble Self Interactions
		obtain_Desired_Operator_Leaf(0.0, 0.0, 1.0, selfInteraction);

	}



	// evaluate multipoles
	void evaluate_multipoles() {
		#pragma omp parallel for		
		for (int k=0; k<nBoxesPerLevel[nLevels]; ++k) {
			tree[nLevels][k].multipoles = Eigen::VectorXd(rank);
			for (int m=0; m<rank; ++m) { // multipoles
				tree[nLevels][k].multipoles[m] = K->f(tree[nLevels][k].chebNodes[m].x, tree[nLevels][k].chebNodes[m].y);
			}
		}
	}




/*	void evaluate_All_M2M() {
		for (int j=nLevels-1; j>1; --j) { // parent
			int J	=	j+1; // children
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) { // parent
				int K	=	4*k; // children
				if (tree[J][K].exists) {
					Eigen::VectorXd child_mutlipoles = Eigen::VectorXd(4*rank);
					child_mutlipoles.segment(0,rank) = tree[J][K].multipoles;
					child_mutlipoles.segment(rank,rank) = tree[J][K+1].multipoles;
					child_mutlipoles.segment(2*rank,rank) = tree[J][K+2].multipoles;
					child_mutlipoles.segment(3*rank,rank) = tree[J][K+3].multipoles;
					tree[j][k].multipoles	=	M2M*child_mutlipoles;
				}
			}
		}
	}

*/




	void evaluate_All_M2M() {
		for (int j=nLevels-1; j>1; --j) { // parent
			//int J	=	j+1; // children
			//#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) { // parent
				//int K	=	4*k; // children
				tree[j][k].multipoles = Eigen::VectorXd(rank);
				for (int m=0; m<rank; ++m) { // multipoles
					tree[j][k].chebNodes = shift_scale_Cheb_Nodes(tree[j][k].center.x, tree[j][k].center.y, boxRadius[j]);
					tree[j][k].multipoles[m] = K->f(tree[j][k].chebNodes[m].x, tree[j][k].chebNodes[m].y);
					//tree[j][k].multipoles	=	M2M[0]*tree[J][K].multipoles+M2M[1]*tree[J][K+1].multipoles+M2M[2]*tree[J][K+2].multipoles+M2M[3]*tree[J][K+3].multipoles;
				}
			}
		}
	}


	void evaluate_M2L() { // list 2
		for (int j=0; j<=1; ++j) {
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				tree[j][k].locals	=	Eigen::VectorXd::Zero(rank);
			}
		}
		for (int j=2; j<=nLevels; ++j) {
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				tree[j][k].locals	=	Eigen::VectorXd::Zero(rank);
				#ifdef HOMOG
				//	Inner well-separated clusters
				for (int l=0; l<16; ++l) {
					int nInner	=	tree[j][k].innerNumbers[l];
					if (nInner>-1) {
						tree[j][k].locals+=M2LInner[l]*tree[j][nInner].multipoles;
					}
				}
				//	Outer well-separated clusters
				for (int l=0; l<24; ++l) {
					int nOuter	=	tree[j][k].outerNumbers[l];
					if (nOuter>-1) {
						tree[j][k].locals+=M2LOuter[l]*tree[j][nOuter].multipoles;
					}
				}
				tree[j][k].locals*=boxRadius[j];					
				#elif LOGHOMOG
				//cout << "yes" << "	j: " << j << "	k: " << k << endl;
				//	Inner well-separated clusters
				for (int l=0; l<16; ++l) {
					int nInner	=	tree[j][k].innerNumbers[l];
					if (nInner>-1) {
						tree[j][k].locals += (1/(2*PI))*M2LInner[l]*tree[j][nInner].multipoles;
						double temp = (1/(2*PI))*boxLogHomogRadius[j]*M2L_term2*tree[j][nInner].multipoles;
						tree[j][k].locals+=temp*Eigen::VectorXd::Ones(rank);
					}
				}
				//	Outer well-separated clusters
				for (int l=0; l<24; ++l) {
					int nOuter	=	tree[j][k].outerNumbers[l];
					if (nOuter>-1) {
						tree[j][k].locals += (1/(2*PI))*M2LOuter[l]*tree[j][nOuter].multipoles;
						double temp = (1/(2*PI))*boxLogHomogRadius[j]*M2L_term2*tree[j][nOuter].multipoles;
						tree[j][k].locals+=temp*Eigen::VectorXd::Ones(rank);
					}
				}
				tree[j][k].locals*=pow(boxRadius[j],2);
				#endif
			}
		}
	}





	void evaluate_All_L2L() {
		for (int j=2; j<nLevels; ++j) {
			int J	=	j+1;
			#pragma omp parallel for
			for (int k=0; k<nBoxesPerLevel[j]; ++k) {
				int K	=	4*k;
				tree[J][K].locals+=L2L[0]*tree[j][k].locals;
				tree[J][K+1].locals+=L2L[1]*tree[j][k].locals;
				tree[J][K+2].locals+=L2L[2]*tree[j][k].locals;
				tree[J][K+3].locals+=L2L[3]*tree[j][k].locals;
			}
		}
	}



//	Self, Neighbor, List1_large, List1_small Interaction
	void evaluate_neighbors() {
		for (int i=0; i<childless_boxes.size(); ++i) { // all childless boxes
			int j = childless_boxes[i].level; // box b
			int k = childless_boxes[i].box;
			#ifdef HOMOG
			//	neighborInteraction
			for (int l=0; l<8; ++l) {
				int nNeighbor	=	tree[j][k].neighborNumbers[l];
				if (nNeighbor>-1) {
					tree[j][k].locals+=boxHomogRadius[j]*neighborInteraction[l]*tree[j][nNeighbor].multipoles;
				}
			}
			//	Self Interaction
			tree[j][k].locals+=boxHomogRadius[j]*selfInteraction*tree[j][k].multipoles;

			#elif LOGHOMOG
			//	neighborInteraction
			for (int l=0; l<8; ++l) {
				int nNeighbor	=	tree[j][k].neighborNumbers[l];
				if (nNeighbor>-1) {
					tree[j][k].locals += pow(boxRadius[j],2)*(1/(2*PI))*neighborInteraction[l]*tree[j][nNeighbor].multipoles;
					double temp = pow(boxRadius[j],2)*(1/(2*PI))*boxLogHomogRadius[j]*M2L_term2*tree[j][nNeighbor].multipoles;
					tree[j][k].locals+=temp*Eigen::VectorXd::Ones(rank);
				}
			}
			//	Self Interaction
			tree[j][k].locals += pow(boxRadius[j],2)*(1/(2*PI))*selfInteraction*tree[j][k].multipoles;
			double temp = pow(boxRadius[j],2)*(1/(2*PI))*boxLogHomogRadius[j]*M2L_term2*tree[j][k].multipoles;
			tree[j][k].locals+=temp*Eigen::VectorXd::Ones(rank);
			#endif
		}
	}



	void perform_Error_Check() {
		srand(time(NULL));
		int nBox	=	rand()%nBoxesPerLevel[nLevels];
		//for (int c=0; c<childless_boxes.size(); ++c) {
		int nNodes = 6;
				
		Eigen::VectorXd potential	=	Eigen::VectorXd::Zero(rank);
		for (int l1=0; l1<rank; ++l1) {// cheb nodes of nBox
			user_function3* mine	=	new user_function3();
			mine->initialise(tree[nLevels][nBox].chebNodes[l1], 0, 0, L, nNodes);
			mine->obtain_Quadrature(epsilon);
			potential(l1) = mine->integral/(2*PI);
		}
		Eigen::VectorXd error(rank);
		for (int k=0; k<rank; ++k) {
			//error(k)	=	fabs(tree[nBox.level][nBox.box].locals(k));
			error(k)	=	fabs((potential-tree[nLevels][nBox].locals)(k)/potential(k));
		}
		cout << " nBox: " << nBox << " er: "<< error.maxCoeff() << endl;
		//cout <<endl << error << endl;
		//}
		
		//return error.maxCoeff();
	}


	void check4() {
		
		cout << "nn" << endl;
		for (int l=0; l<8; ++l) {
			cout << tree[2][0].neighborNumbers[l]<<",";
		}
		cout << endl << "in" << endl;
		for (int l=0; l<16; ++l) {
			cout << tree[2][0].innerNumbers[l]<<",";
		}
		cout << endl << "on" << endl;
		for (int l=0; l<16; ++l) {
			cout << tree[2][0].outerNumbers[l]<<",";
		}
		cout << endl;
	}

};

#endif
