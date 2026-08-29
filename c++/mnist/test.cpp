#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <bits/stdc++.h>
#include <random>
#include <iostream>
#include <fstream>
#include <armadillo>
#include <fstream>
#include "omp.h"

using namespace std;
using namespace arma;

int main (void) {

	auto t0 = chrono::high_resolution_clock::now();

	int d = 784, K = 10;
	int length_eps = 1;
    	double eps[length_eps] = {200};
    	double threshold = 10;
    	double delta = 0.5;
    	
    	int n_rows = 28;
    	int n_cols = 28;
    	int n_images = 60000;

	int n_iter_N = 1;
	int length_N = 1;
	int N[length_N] = {60000};
		
    	srand(time(0));
	random_device rd;
	mt19937 gen(rd());
	default_random_engine generator;
    	
    	vector<int> time_standard_Kmeans(length_N, 0);
    	vector<vector<int>> time_EKMeans(length_N, vector<int>(length_eps, 0));
    	
    	vector<int> time_EKMeans_preprocessing(length_N, 0);

    	
    	vector<int> iterations_standard_Kmeans(length_N, 0);
    	vector<vector<int>> iterations_EKMeans(length_N, vector<int>(length_eps, 0));
  	   	
    	vector<vector<double>> RSS_EKMeans_Lloyds(length_N, vector<double>(length_eps, 0));
    	


    	 	
    	 	
    	// Read MNIST dataset
    	vector<vector<double>> V(n_images, vector<double>(d));
	mat V_aux(n_images, d);
    	
    	ifstream file ("t10k-images.idx3-ubyte", ios::binary);
    	for (int i = 0; i < n_images; ++i) {
		for (int r = 0; r < n_rows; ++r) {
	        	for (int c = 0; c < n_cols; ++c) {
	            		unsigned char temp = 0;
	            		file.read((char*)&temp,sizeof(temp));
	            		V[i][(n_rows*r)+c] = ((double)temp);
	            		V_aux(i,(n_rows*r)+c) = ((double)temp);
	        	}
	    	}
	}
        file.close();
    	
    	for (int n = 0; n < length_N; n++) {
    	
  		uniform_int_distribution<int> unif_distribution(0, N[n]-1);
  		
  		for (int iter_N = 0; iter_N < n_iter_N; iter_N++) {   	    		
	    			    		
	    		//------------------------ EKMEANS PRE-PROCESSING ------------------------//
	    		
	    		// Initial time
	    		auto begin = chrono::high_resolution_clock::now();
	    		
	    		// Compute the norms
	    		double spectral_norm = norm2est(V_aux);
	    		vector<double> V_norms(N[n], 0);
	    		double V21_norm = 0;
	    		for (int i = 0; i < N[n]; i++) {
	    			for (int h = 0; h < d; h++)
	    				V_norms[i] += V[i][h] * V[i][h];

	    			V_norms[i] = sqrt(V_norms[i]);
	    			V21_norm += V_norms[i];
	    		}
	    		
	    		// Create sampling distribution for EKMeans
		    	vector<double> Q_distr(N[n]);
		    	for (int i = 0; i < N[n]; i++)
				Q_distr[i] = V_norms[i] / V21_norm;
				
			discrete_distribution<> Q_distribution(Q_distr.begin(), Q_distr.end());
			
			// Final time
			auto end = chrono::high_resolution_clock::now();
		    	time_EKMeans_preprocessing[n] += chrono::duration_cast<chrono::milliseconds>(end - begin).count();
			
								
/////////////////////////////////////////////////////////////////////////////////////////////////////////	    			    		
////////////////////// LLOYD'S ALGORITHM ////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////	
		    		
	    		begin = chrono::high_resolution_clock::now();
	    	
	    		int list[K] = {10840,56267,14849,2726,47180,1640,52730,21847,20394,45806};
	    		// Sample initial centroids
		    	vector<vector<double>> C_standard(K, vector<double>(d));
		    	for (int i = 0; i < K; i++) {
		    		for (int h = 0; h < d; h++) 
		    			C_standard[i][h] = V[list[i]][h];
		    	}
			
			// Main Lloyd's loop
			double distance_centroids = 2 * K * threshold;
			double iterations = 0;
			while (distance_centroids > K * threshold and iterations < 50) {
				
			    	// Compute cluster sizes
			    	vector<int> C_size(K, 0);
			    	for (int i = 0; i < N[n]; i++) {
			    	
					int min_index = -1;
				    	double min_distance = DBL_MAX;
				    	for (int j = 0; j < K; j++) {
				    	
						double new_dist = 0;
						for (int h = 0; h < d; h++) 
					    		new_dist += (C_standard[j][h] - V[i][h]) * (C_standard[j][h] - V[i][h]);
				 
						if (new_dist < min_distance) {
					    		min_distance = new_dist;
					    		min_index = j;
						}
				    	}
					C_size[min_index]++;
			    	}
			    	
			    	// Compute new centroids
			    	vector<vector<double>> C_new(K, vector<double>(d,0));
			    	for (int i = 0; i < N[n]; i++) {
			    	
			    		int min_index = -1;
				    	double min_distance = DBL_MAX;
				    	for (int j = 0; j < K; j++) {
				    	
						double new_dist = 0;
						for (int h = 0; h < d; h++) 
					    		new_dist += (C_standard[j][h] - V[i][h]) * (C_standard[j][h] - V[i][h]);
				 
						if (new_dist < min_distance) {
					    		min_distance = new_dist;
					    		min_index = j;
						}
				    	}

					for (int h = 0; h < d; h++)  
				    		C_new[min_index][h] += V[i][h];
			    	}
			    	
			    	for (int j = 0; j < K; j++) {
			    		if (C_size[j] == 0) {
			    			for (int h = 0; h < d; h++)
			    				C_new[j][h] = 0;
			    		}
			    		else {
			    			for (int h = 0; h < d; h++)
			    				C_new[j][h] = C_new[j][h] / C_size[j];
			    		}
			    	}
			    	
			    	// Compute distance between centroids
			    	distance_centroids = 0;	
			    	for (int j = 0; j < K; j++) {
			    	
			    		double distance_aux = 0;
					for (int h = 0; h < d; h++) 
				    		distance_aux += abs((C_standard[j][h] - C_new[j][h]) * (C_standard[j][h] - C_new[j][h]));
				  	distance_centroids += sqrt(abs(distance_aux));
			    	}
			  		
			  	for (int j = 0; j < K; j++) {
			  		for (int h = 0; h < d; h++)
			    			C_standard[j][h] = C_new[j][h];
			    	}			    	
			    		
			    	iterations += 1;
			    	cout << distance_centroids << endl;
			}
			iterations_standard_Kmeans[n] += iterations;

		    	// Compute duration
		    	end = chrono::high_resolution_clock::now();
		    	time_standard_Kmeans[n] += chrono::duration_cast<chrono::milliseconds>(end - begin).count();
		//    	cout << "Iterations standard: " << iterations << endl;		    					   
			cout << endl;
			

/////////////////////////////////////////////////////////////////////////////////////////////////////////	    				    		
/////////////////////// EKMEANS /////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
	   						
			for (int e = 0; e < length_eps; e++) {
			
			    	// Initial time
			    	begin = chrono::high_resolution_clock::now();
			
		    		long int p = ceil( spectral_norm * spectral_norm / N[n] * K * K / (eps[e] * eps[e]) * log(K/delta) );
		    		long int q = ceil( (V21_norm / N[n])  * (V21_norm / N[n]) * K * K / (eps[e] * eps[e]) * log(K/delta) );		    		
		    		
//				cout << "p: " << p << endl;
//				cout << "q: " << q << endl;
			    	
				// Sample p vectors
				vector<vector<double>> V_P_EKMeans(p, vector<double>(d));
			    	for (int i = 0; i < p; i++) {
			    		int j = unif_distribution(generator);
			    		for (int h = 0; h < d; h++) 
			 			V_P_EKMeans[i][h] = V[j][h];
				}
				
				// Sample q vectors
				vector<vector<double>> V_Q_EKMeans(q, vector<double>(d));
				vector<vector<double>> V_Q_EKMeans_normalised(q, vector<double>(d));
				for (int i = 0; i < q; i++) {
					int j = Q_distribution(gen);
					
					for (int h = 0; h < d; h++) {
						V_Q_EKMeans[i][h] = V[j][h];
						V_Q_EKMeans_normalised[i][h] = V[j][h] / V_norms[j];
					}
				}
				
				// Sample initial centroids
			    	vector<vector<double>> C_EKMeans(K, vector<double>(d));
				for (int i = 0; i < K; i++) {
					for (int h = 0; h < d; h++) 
		    				C_EKMeans[i][h] = V[list[i]][h];
		    		}
												    			 				    		
					
			    	// Main EKK-means loop
			    	distance_centroids = 2 * K * threshold;
			    	iterations = 0.;
				while (distance_centroids > K * threshold and iterations < 50) {
//				while (distance_centroids > K * threshold) {
				
				    	// Compute cluster sizes
				    	vector<int> C_size(K, 0);
				    	for (int i = 0; i < p; i++) {
				    	
						int min_index = -1;
					    	double min_distance = DBL_MAX;
					    	for (int j = 0; j < K; j++) {
					    	
							double new_dist = 0;
							for (int h = 0; h < d; h++) 
						    		new_dist += (C_EKMeans[j][h] - V_P_EKMeans[i][h]) * (C_EKMeans[j][h] - V_P_EKMeans[i][h]);
					 
							if (new_dist < min_distance) {
						    		min_distance = new_dist;
						    		min_index = j;
							}
					    	}
						C_size[min_index]++;
				    	}
				    	
				    	
				    	// Compute new centroids
				    	vector<vector<double>> C_new2(K, vector<double>(d, 0));
				    	for (int i = 0; i < q; i++) {
				    	
				    		int min_index = -1;
					    	double min_distance = DBL_MAX;
					    	for (int j = 0; j < K; j++) {
					    	
							double new_dist = 0;
							for (int h = 0; h < d; h++) 
						    		new_dist += (C_EKMeans[j][h] - V_Q_EKMeans[i][h]) * (C_EKMeans[j][h] - V_Q_EKMeans[i][h]);
					 
							if (new_dist < min_distance) {
						    		min_distance = new_dist;
						    		min_index = j;
							}
					    	}

						for (int h = 0; h < d; h++)
					    		C_new2[min_index][h] += V_Q_EKMeans_normalised[i][h];
				    	}
				    	
				    	for (int j = 0; j < K; j++) {
				    		if (C_size[j] == 0) 
				    			C_new2[j] = C_EKMeans[j];			    
				    		else {
				    			double coeff = (((V21_norm / N[n]) / q) * p) / C_size[j];
				    			for (int h = 0; h < d; h++)
				    				C_new2[j][h] = coeff * C_new2[j][h];
				    		}
				    	}
				    	
				    	// Compute distance between centroids
				    	distance_centroids = 0;	
				    	for (int j = 0; j < K; j++) {
				    	
				    		double distance_aux = 0;
						for (int h = 0; h < d; h++) 
					    		distance_aux += abs((C_EKMeans[j][h] - C_new2[j][h]) * (C_EKMeans[j][h] - C_new2[j][h]));
					  	distance_centroids += sqrt(abs(distance_aux));
				    	}				    

					for (int j = 0; j < K; j++) {
			  			for (int h = 0; h < d; h++)
			    				C_EKMeans[j][h] = C_new2[j][h];
			    		}
				    		
				    	iterations += 1;	
				    	cout << distance_centroids << endl;				 
				}					
				iterations_EKMeans[n][e] += iterations;
				
				// Compute duration
			    	end = chrono::high_resolution_clock::now();
			    	time_EKMeans[n][e] += chrono::duration_cast<chrono::milliseconds>(end - begin).count();
		
		//		cout << "Iterations EKmeans: " << iterations << endl;
				cout << endl;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// POS-PROCESSING //////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
			    	
			    	// Compute RSS
			    	long double RSS_standard_aux = 0., RSS_EKMeans_aux = 0.;
			    	
			    	for (int i = 0; i < N[n]; i++) {
			    	
				    	long double min_distance_standard = DBL_MAX, min_distance_EKMeans = DBL_MAX;
				    	int min_index_standard, min_index_EKMeans, min_index_mini_batch, min_index_coresets;
				    	for (int j = 0; j < K; j++) {
				    	
						long double new_dist_standard = 0., new_dist_EKMeans = 0.;
						for (int h = 0; h < d; h++) {
							new_dist_standard += (C_standard[j][h] - V[i][h]) * (C_standard[j][h] - V[i][h]);
					    		new_dist_EKMeans += (C_EKMeans[j][h] - V[i][h]) * (C_EKMeans[j][h] - V[i][h]);					    	
					    	}
				 
						if (new_dist_standard < min_distance_standard) {
					    		min_distance_standard = new_dist_standard;
					    		min_index_standard = j;
					    	}
					    	if (new_dist_EKMeans < min_distance_EKMeans) {
					    		min_distance_EKMeans = new_dist_EKMeans;
					    		min_index_EKMeans = j;
					    	}					    	
				    	}
				    	RSS_standard_aux += min_distance_standard;
				    	RSS_EKMeans_aux += min_distance_EKMeans;				    	
			    	}
			    				    	
			    			
			    	RSS_EKMeans_Lloyds[n][e] += 100. * (RSS_EKMeans_aux - RSS_standard_aux) / RSS_standard_aux;
		//	    	cout << "RSS EKMeans vs Lloyds: " << RSS_EKMeans_Lloyds[n][e] << endl;
			    	
		
			}
			cout << n << ' ' << iter_N << endl; 						
		}
    	}
    	
    	// Write the output into a separate file
	ofstream MyFile;
	
	MyFile.open("EKmeans_vs_Lloyds_test.txt");
	MyFile << "n  eps  time  iterations  RSS" << endl;
	MyFile << "samples" << ' ' << n_iter_N << endl;
	
	for (int n = 0; n < length_N; n++) {
		MyFile << N[n] << ' ' << 0.0 << ' ' << time_standard_Kmeans[n] / ((double) n_iter_N) << ' ' << iterations_standard_Kmeans[n] / ((double) n_iter_N) << endl;
		cout << "N: " << N[n] << endl;
		cout << "Milliseconds standard: " << time_standard_Kmeans[n] / ((double) n_iter_N) << endl;
    		cout << "Iterations standard: " << iterations_standard_Kmeans[n] / ((double) n_iter_N) << endl << endl;
    		
		for (int e = 0; e < length_eps; e++) {
			MyFile << N[n] << ' ' << eps[e] << ' ' << time_EKMeans[n][e] / ((double) n_iter_N) << ' ' << iterations_EKMeans[n][e] / ((double) n_iter_N) << ' ' << RSS_EKMeans_Lloyds[n][e] << endl;
			cout << "Miliseconds EKmeans: " << time_EKMeans[n][e] / ((double) n_iter_N) << endl;
    			cout << "Iterations EKmeans: " << iterations_EKMeans[n][e] / ((double) n_iter_N) << endl;
    			cout << "RSS EKmeans vs Lloyds (%): " << RSS_EKMeans_Lloyds[n][e] << endl;   			
    		}
	}
	MyFile.close();	

	auto t1 = chrono::high_resolution_clock::now();
	cout <<  chrono::duration_cast<chrono::seconds>(t1 - t0).count() << endl;
    	
    	return 0;

}
