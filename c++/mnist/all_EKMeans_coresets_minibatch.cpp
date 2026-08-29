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
    //	double eps[length_eps] = {0.5, 1., 1.5, 2., 2.5};
   	double eps[length_eps] = {2.5};
    	double threshold = 0.15;
    	double delta = 0.01;
    	
    	int n_rows = 28;
    	int n_cols = 28;
    	int n_images = 60000;

	int n_iter_N = 1;
	int length_N = 1;
	//int N[length_N] = {20000, 25000, 30000, 35000, 40000, 45000, 50000};
	int N[length_N] = {20000};
		
    	srand(time(0));
	random_device rd;
	mt19937 gen(rd());
	default_random_engine generator;
    	
    	vector<int> time_standard_Kmeans(length_N, 0);
    	vector<vector<int>> time_EKMeans(length_N, vector<int>(length_eps, 0));
    	vector<vector<int>> time_mini_batch(length_N, vector<int>(length_eps, 0));
    	vector<vector<int>> time_coresets(length_N, vector<int>(length_eps, 0));
    	
    	vector<int> time_EKMeans_preprocessing(length_N, 0);
    	vector<int> time_coresets_preprocessing(length_N, 0);
    	
    	vector<int> iterations_standard_Kmeans(length_N, 0);
    	vector<vector<int>> iterations_EKMeans(length_N, vector<int>(length_eps, 0));
    	vector<vector<int>> iterations_mini_batch(length_N, vector<int>(length_eps, 0));
    	vector<vector<int>> iterations_coresets(length_N, vector<int>(length_eps, 0));
    	   	
    	vector<vector<double>> RSS_EKMeans_Lloyds(length_N, vector<double>(length_eps, 0));
    	vector<vector<double>> ARI_EKMeans_Lloyds(length_N, vector<double>(length_eps, 0));
    	vector<vector<double>> NMI_EKMeans_Lloyds(length_N, vector<double>(length_eps, 0));
    	
    	vector<vector<double>> RSS_mini_batch_Lloyds(length_N, vector<double>(length_eps, 0));
    	vector<vector<double>> ARI_mini_batch_Lloyds(length_N, vector<double>(length_eps, 0));
    	vector<vector<double>> NMI_mini_batch_Lloyds(length_N, vector<double>(length_eps, 0));
    	
    	vector<vector<double>> RSS_coresets_Lloyds(length_N, vector<double>(length_eps, 0));
    	vector<vector<double>> ARI_coresets_Lloyds(length_N, vector<double>(length_eps, 0));
    	vector<vector<double>> NMI_coresets_Lloyds(length_N, vector<double>(length_eps, 0));
    	
    	vector<vector<double>> RSS_EKMeans_coresets(length_N, vector<double>(length_eps, 0));
    	vector<vector<double>> ARI_EKMeans_coresets(length_N, vector<double>(length_eps, 0));
    	vector<vector<double>> NMI_EKMeans_coresets(length_N, vector<double>(length_eps, 0));
    	 	
    	 	
    	// Read MNIST dataset
    	vector<vector<double>> mnist_dataset(n_images, vector<double>(d));
    	
    	ifstream file ("t10k-images.idx3-ubyte", ios::binary);
    	for (int i = 0; i < n_images; ++i) {
		for (int r = 0; r < n_rows; ++r) {
	        	for (int c = 0; c < n_cols; ++c) {
	            		unsigned char temp = 0;
	            		file.read((char*)&temp,sizeof(temp));
	            		mnist_dataset[i][(n_rows*r)+c] = ((double)temp) / 255;
	        	}
	    	}
	}
        file.close();
    	
    	for (int n = 0; n < length_N; n++) {
    	
  		uniform_int_distribution<int> unif_distribution(0, N[n]-1);
  		
  		for (int iter_N = 0; iter_N < n_iter_N; iter_N++) {
    	
	    		vector<vector<double>> V(N[n], vector<double>(d));
	    		mat V_aux(N[n], d);
	  
	  		// Create the input
	  		unordered_set<int> elems;
			for (int r = n_images - N[n]; r < n_images; ++r) {
				int v = uniform_int_distribution<>(0, r)(gen);
				
				if (!elems.insert(v).second) 
				    elems.insert(r);
			}
	  		
	  		int counter = 0;
	    		for (auto x : elems) {
	    			for (int h = 0 ; h < d; h++) {
	    				V[counter][h] = mnist_dataset[x][h];
	    				V_aux(counter, h) = mnist_dataset[x][h];
	    			}
	    			counter++;
	    		}
	    			    		
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
			
			
			//------------------------ CORESETS PRE-PROCESSING ------------------------//
			
			// Initial time
			begin = chrono::high_resolution_clock::now();
			
			// Using k-means++ as an initial solution
		    	vector<vector<double>> C_coresets_preprocessing(K, vector<double>(d));
		    	vector<double> distances_coresets(N[n], 0);
		    	
		    	int a = unif_distribution(generator);
		    	int threshold_sampling = 0;
		    	
			C_coresets_preprocessing[0] = V[a];
			for (int i = 0; i < N[n]; i++) {	
	    			for (int h = 0; h < d; h++)
	    				distances_coresets[i] += (C_coresets_preprocessing[0][h] - V[i][h]) * (C_coresets_preprocessing[0][h] - V[i][h]);
	    				
	    			threshold_sampling += distances_coresets[i];
	    		}
	    		uniform_real_distribution<double> real_distribution_coresets(0.0, threshold_sampling);
	    		int random_number = real_distribution_coresets(generator);
	    		int cumulative = 0;
	    		a = 0;
	    		while (cumulative < random_number) {
	    			cumulative += distances_coresets[a];
	    			a++;
	    		}
	    		C_coresets_preprocessing[1] = V[a-1];
			
		    	for (int k = 2; k < K; k++) {
		    		for (int i = 0; i < N[n]; i++) {
		    		
		    			double dist_aux = 0;
		    			for (int h = 0; h < d; h++)
		    				dist_aux += (C_coresets_preprocessing[k-1][h] - V[i][h]) * (C_coresets_preprocessing[k-1][h] - V[i][h]);
		    				
		    			if (dist_aux < distances_coresets[i]) {
		    				threshold_sampling += (dist_aux - distances_coresets[i]);
		    				distances_coresets[i] = dist_aux;
		    			}
		    		}
		    		uniform_real_distribution<double> real_distribution_coresets(0.0, threshold_sampling);
		    		random_number = real_distribution_coresets(generator);
		    		cumulative = 0;
		    		a = 0;
		    		while (cumulative < random_number) {
	    				cumulative += distances_coresets[a];
	    				a++;
	    			}
	    			C_coresets_preprocessing[k] = V[a-1];
			}
			
			// Compute warm start cluster sizes and costs
		    	vector<int> C_size_preprocessing(K, 0);
		    	vector<int> C_pointers_preprocessing(N[n]);
		    	vector<double> RSS_clusters_preprocessing(K, 0);
		    	vector<double> RSS_vectors_preprocessing(N[n], 0);
		    	double RSS_total_preprocessing = 0.;
		    	
		    	for (int i = 0; i < N[n]; i++) {
		    	
				int min_index = -1;
			    	double min_distance = DBL_MAX;
			    	for (int j = 0; j < K; j++) {
			    	
					double new_dist = 0;
					for (int h = 0; h < d; h++) 
				    		new_dist += (C_coresets_preprocessing[j][h] - V[i][h]) * (C_coresets_preprocessing[j][h] - V[i][h]);
			 
					if (new_dist < min_distance) {
				    		min_distance = new_dist;
				    		min_index = j;
					}
			    	}
				C_size_preprocessing[min_index]++;
				C_pointers_preprocessing[i] = min_index;
				RSS_clusters_preprocessing[min_index] += min_distance;
				RSS_vectors_preprocessing[i] = min_distance;
				RSS_total_preprocessing += min_distance;
		    	}
		    	
		    	// Create sampling distribution for coresets
		    	vector<double> coresets_distr(N[n]);
		    	for (int i = 0; i < N[n]; i++) {
		    		int k = C_pointers_preprocessing[i];
		    		coresets_distr[i] = 0.25 * (1. / (K * C_size_preprocessing[k]) + RSS_vectors_preprocessing[i] / (K * RSS_clusters_preprocessing[k]) + RSS_vectors_preprocessing[i] / RSS_total_preprocessing + RSS_clusters_preprocessing[k] / RSS_total_preprocessing / C_size_preprocessing[k] );
		    	}
			
			discrete_distribution<> coresets_distribution(coresets_distr.begin(), coresets_distr.end());
			
			// Final time
			end = chrono::high_resolution_clock::now();
		    	time_coresets_preprocessing[n] += chrono::duration_cast<chrono::milliseconds>(end - begin).count();
			
			
/////////////////////////////////////////////////////////////////////////////////////////////////////////	    			    		
////////////////////// LLOYD'S ALGORITHM ////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////	
		    		
	    		begin = chrono::high_resolution_clock::now();
	    		
	    		// Sample initial centroids
		    	vector<vector<double>> C_initial(K, vector<double>(d));
			vector<vector<double>> C_standard(K, vector<double>(d));
		    	for (int i = 0; i < K; i++) {
		    		int j = unif_distribution(generator);
				C_initial[i] = V[j];
			}
			C_standard = C_initial;
							    				
			
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
			  				    	
			    	C_standard = C_new;
			    	iterations += 1;
			}
			iterations_standard_Kmeans[n] += iterations;

		    	// Compute duration
		    	end = chrono::high_resolution_clock::now();
		    	time_standard_Kmeans[n] += chrono::duration_cast<chrono::milliseconds>(end - begin).count();
		//    	cout << "Iterations standard: " << iterations << endl;		    					   


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
			 		V_P_EKMeans[i] = V[j];
				}
				
				// Sample q vectors
				vector<vector<double>> V_Q_EKMeans(q, vector<double>(d));
				vector<vector<double>> V_Q_EKMeans_normalised(q, vector<double>(d));
				for (int i = 0; i < q; i++) {
					int j = Q_distribution(gen);
					V_Q_EKMeans[i] = V[j];
					for (int l = 0; l < d; l++)
						V_Q_EKMeans_normalised[i][l] = V[j][l] / V_norms[j];
				}
				
				// Sample initial centroids
			    	vector<vector<double>> C_EKMeans(K, vector<double>(d));
				C_EKMeans = C_initial;
												    			 				    		
					
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
				    	vector<vector<double>> C_new(K, vector<double>(d, 0));
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
					    		C_new[min_index][h] += V_Q_EKMeans_normalised[i][h];
				    	}
				    	
				    	for (int j = 0; j < K; j++) {
				    		if (C_size[j] == 0) 
				    			C_new[j] = C_EKMeans[j];			    
				    		else {
				    			double coeff = (((V21_norm / N[n]) / q) * p) / C_size[j];
				    			for (int h = 0; h < d; h++)
				    				C_new[j][h] = coeff * C_new[j][h];
				    		}
				    	}
				    	
				    	// Compute distance between centroids
				    	distance_centroids = 0;	
				    	for (int j = 0; j < K; j++) {
				    	
				    		double distance_aux = 0;
						for (int h = 0; h < d; h++) 
					    		distance_aux += abs((C_EKMeans[j][h] - C_new[j][h]) * (C_EKMeans[j][h] - C_new[j][h]));
					  	distance_centroids += sqrt(abs(distance_aux));
				    	}				    

				    	C_EKMeans = C_new;
				    	iterations += 1;					 
				}					
				iterations_EKMeans[n][e] += iterations;
				
				// Compute duration
			    	end = chrono::high_resolution_clock::now();
			    	time_EKMeans[n][e] += chrono::duration_cast<chrono::milliseconds>(end - begin).count();
		
		//		cout << "Iterations EKmeans: " << iterations << endl;

				
/////////////////////////////////////////////////////////////////////////////////////////////////////////									
/////////////////////////////// MINI-BATCH //////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////			    	
			    	
			    	// Initial time
			    	begin = chrono::high_resolution_clock::now();
				
				p = ceil( spectral_norm * spectral_norm / N[n] * K * K / (eps[e] * eps[e]) * log(K/delta) );
		    		q = ceil( (V21_norm / N[n])  * (V21_norm / N[n]) * K * K / (eps[e] * eps[e]) * log(K/delta) );		    
			    	
				// Sample p vectors
				vector<vector<double>> V_P_mini_batch(p, vector<double>(d));
			    	for (int i = 0; i < p; i++) {
			    		int j = unif_distribution(generator);
			 		V_P_mini_batch[i] = V[j];
				}
				
				// Sample q vectors mini_batch
				vector<vector<double>> V_Q_mini_batch(q, vector<double>(d));
				for (int i = 0; i < q; i++) {
					int j = unif_distribution(generator);
			 		V_Q_mini_batch[i] = V[j];
				}
				
				// Sample initial centroids
			    	vector<vector<double>> C_mini_batch(K, vector<double>(d));
			    	C_mini_batch = C_initial;				
								
				// Main mini batch loop
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
						    		new_dist += (C_mini_batch[j][h] - V_P_mini_batch[i][h]) * (C_mini_batch[j][h] - V_P_mini_batch[i][h]);
					 
							if (new_dist < min_distance) {
						    		min_distance = new_dist;
						    		min_index = j;
							}
					    	}
						C_size[min_index]++;
				    	}
				    	
				    	
				    	// Compute new centroids
				    	vector<vector<double>> C_new(K, vector<double>(d, 0));
				    	for (int i = 0; i < q; i++) {
				    	
				    		int min_index = -1;
					    	double min_distance = DBL_MAX;
					    	for (int j = 0; j < K; j++) {
					    	
							double new_dist = 0;
							for (int h = 0; h < d; h++) 
						    		new_dist += (C_mini_batch[j][h] - V_Q_mini_batch[i][h]) * (C_mini_batch[j][h] - V_Q_mini_batch[i][h]);
					 
							if (new_dist < min_distance) {
						    		min_distance = new_dist;
						    		min_index = j;
							}
					    	}
					    	
					    	for (int h = 0; h < d; h++)
					    		C_new[min_index][h] += V_Q_mini_batch[i][h];						
				    	}
				    	
				    	for (int j = 0; j < K; j++) {
				    		if (C_size[j] == 0) 
				    			C_new[j] = C_mini_batch[j];			    
				    		else {
				    			double coeff = ((double) p) / q / C_size[j];
				    			for (int h = 0; h < d; h++)
				    				C_new[j][h] = coeff * C_new[j][h];
				    		}
				    	}
				    	
				    	// Compute distance between centroids
				    	distance_centroids = 0;	
				    	for (int j = 0; j < K; j++) {
				    	
				    		double distance_aux = 0;
						for (int h = 0; h < d; h++) 
					    		distance_aux += abs((C_mini_batch[j][h] - C_new[j][h]) * (C_mini_batch[j][h] - C_new[j][h]));
					  	distance_centroids += sqrt(abs(distance_aux));
				    	}				    

				    	C_mini_batch = C_new;
				    	iterations += 1;					 
				}					
				iterations_mini_batch[n][e] += iterations;					

			    	// Compute duration
			    	end = chrono::high_resolution_clock::now();
			    	time_mini_batch[n][e] += chrono::duration_cast<chrono::milliseconds>(end - begin).count();
			    	
		//	    	cout << "Iterations mini batch: " << iterations << endl;


/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// CORESETS ////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

			//------------------------ CORESETS PRE-PROCESSING 2 ------------------------//
				
				int m = (p + q)/2;		    				    						
				
				// Sample m=(p+q)/2 vectors for coresets
				vector<vector<double>> V_coresets(m, vector<double>(d));
				vector<double> weights_coresets(m);
				double weights_coresets_total = 0.;
				
				for (int i = 0; i < m; i++) {
				
					int j = coresets_distribution(gen);
					weights_coresets[i] = sqrt(1. / m / coresets_distr[j]);
					weights_coresets_total += weights_coresets[i];
					V_coresets[i] = V[j];
				}
				
			//------------------------ CORESETS MAIN PROCESSING ------------------------//
				
				// Initial time
			    	begin = chrono::high_resolution_clock::now();
			    				    	
			    	// Sample initial centroids
			    	vector<vector<double>> C_coresets(K, vector<double>(d));
			    	C_coresets = C_initial;
			    				    								
				// Main (weighted) Lloyd's loop
				distance_centroids = 2 * K * threshold;
				iterations = 0;
				while (distance_centroids > K * threshold and iterations < 50) {
					
				    	// Compute cluster sizes
				    	vector<double> C_size(K, 0);
				    	for (int i = 0; i < m; i++) {
				    	
						int min_index = -1;
					    	double min_distance = DBL_MAX;
					    	for (int j = 0; j < K; j++) {
					    	
							double new_dist = 0;
							for (int h = 0; h < d; h++) 
						    		new_dist += (C_coresets[j][h] - V_coresets[i][h]) * (C_coresets[j][h] - V_coresets[i][h]);
					 
							if (new_dist < min_distance) {
						    		min_distance = new_dist;
						    		min_index = j;
							}
					    	}
						C_size[min_index] += weights_coresets[i];
				    	}
				    	
				    	// Compute new centroids
				    	vector<vector<double>> C_new(K, vector<double>(d,0));
				    	for (int i = 0; i < m; i++) {
				    	
				    		int min_index = -1;
					    	double min_distance = DBL_MAX;
					    	for (int j = 0; j < K; j++) {
					    	
							double new_dist = 0;
							for (int h = 0; h < d; h++) 
						    		new_dist += (C_coresets[j][h] - V_coresets[i][h]) * (C_coresets[j][h] - V_coresets[i][h]);
					 
							if (new_dist < min_distance) {
						    		min_distance = new_dist;
						    		min_index = j;
							}
					    	}

						for (int h = 0; h < d; h++)  
					    		C_new[min_index][h] += weights_coresets[i] * V_coresets[i][h];
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
					    		distance_aux += abs((C_coresets[j][h] - C_new[j][h]) * (C_coresets[j][h] - C_new[j][h]));
					  	distance_centroids += sqrt(abs(distance_aux));
				    	}
				  				    	
				    	C_coresets = C_new;
				    	iterations += 1;
				}
				iterations_coresets[n][e] += iterations;

			    	// Compute duration
			    	end = chrono::high_resolution_clock::now();
			    	time_coresets[n][e] += chrono::duration_cast<chrono::milliseconds>(end - begin).count();
			//    	cout << "Iterations standard: " << iterations << endl;	


/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// POS-PROCESSING //////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
			    	
			    	// Compute RSS, ARI, NMI
			    	long double RSS_standard_aux = 0., RSS_EKMeans_aux = 0., RSS_mini_batch_aux = 0., RSS_coresets_aux = 0.;
			    	vector<vector<double>> overlap_matrix_EKMeans_Lloyds(K, vector<double>(K, 0));
			    	vector<vector<double>> overlap_matrix_mini_batch_Lloyds(K, vector<double>(K, 0));
			    	vector<vector<double>> overlap_matrix_coresets_Lloyds(K, vector<double>(K, 0));
			    	vector<vector<double>> overlap_matrix_EKMeans_coresets(K, vector<double>(K, 0));
			    	
			    	for (int i = 0; i < N[n]; i++) {
			    	
				    	long double min_distance_standard = DBL_MAX, min_distance_EKMeans = DBL_MAX, min_distance_mini_batch = DBL_MAX, min_distance_coresets = DBL_MAX;
				    	int min_index_standard, min_index_EKMeans, min_index_mini_batch, min_index_coresets;
				    	for (int j = 0; j < K; j++) {
				    	
						long double new_dist_standard = 0., new_dist_EKMeans = 0., new_dist_mini_batch = 0., new_dist_coresets = 0.;
						for (int h = 0; h < d; h++) {
							new_dist_standard += (C_standard[j][h] - V[i][h]) * (C_standard[j][h] - V[i][h]);
					    		new_dist_EKMeans += (C_EKMeans[j][h] - V[i][h]) * (C_EKMeans[j][h] - V[i][h]);
					    		new_dist_mini_batch += (C_mini_batch[j][h] - V[i][h]) * (C_mini_batch[j][h] - V[i][h]);
					    		new_dist_coresets += (C_coresets[j][h] - V[i][h]) * (C_coresets[j][h] - V[i][h]);
					    	}
				 
						if (new_dist_standard < min_distance_standard) {
					    		min_distance_standard = new_dist_standard;
					    		min_index_standard = j;
					    	}
					    	if (new_dist_EKMeans < min_distance_EKMeans) {
					    		min_distance_EKMeans = new_dist_EKMeans;
					    		min_index_EKMeans = j;
					    	}
					    	if (new_dist_mini_batch < min_distance_mini_batch) {
					    		min_distance_mini_batch = new_dist_mini_batch;
					    		min_index_mini_batch = j;
					    	}
					    	if (new_dist_coresets < min_distance_coresets) {
					    		min_distance_coresets = new_dist_coresets;
					    		min_index_coresets = j;
					    	}
				    	}
				    	RSS_standard_aux += min_distance_standard;
				    	RSS_EKMeans_aux += min_distance_EKMeans;
				    	RSS_mini_batch_aux += min_distance_mini_batch;
				    	RSS_coresets_aux += min_distance_coresets;
				    	overlap_matrix_EKMeans_Lloyds[min_index_standard][min_index_EKMeans] += 1;
				    	overlap_matrix_mini_batch_Lloyds[min_index_standard][min_index_mini_batch] += 1;
				    	overlap_matrix_coresets_Lloyds[min_index_standard][min_index_coresets] += 1;
				    	overlap_matrix_EKMeans_coresets[min_index_coresets][min_index_EKMeans] += 1;
			    	}
			    	
			    	vector<double> marginal_distribution_standard(K, 0);
			    	vector<double> marginal_distribution_EKMeans(K, 0);
			    	vector<double> marginal_distribution_mini_batch(K, 0);
			    	vector<double> marginal_distribution_coresets(K, 0);
			    	for (int i = 0; i < K; i++) {
			    		for (int j = 0; j < K; j++) {
			    		     	marginal_distribution_standard[i] += overlap_matrix_EKMeans_Lloyds[i][j];
			    		     	marginal_distribution_EKMeans[i] += overlap_matrix_EKMeans_Lloyds[j][i];
			    		     	marginal_distribution_mini_batch[i] += overlap_matrix_mini_batch_Lloyds[j][i];
			    		     	marginal_distribution_coresets[i] += overlap_matrix_coresets_Lloyds[j][i];		    		     	
			    		}
			    	}
			    			
			    	RSS_EKMeans_Lloyds[n][e] += 100. * (RSS_EKMeans_aux - RSS_standard_aux) / RSS_standard_aux / n_iter_N;
		//	    	cout << "RSS EKMeans vs Lloyds: " << RSS_EKMeans_Lloyds[n][e] << endl;
			    	
			    	RSS_mini_batch_Lloyds[n][e] += 100. * (RSS_mini_batch_aux - RSS_standard_aux) / RSS_standard_aux / n_iter_N;
		//	    	cout << "RSS mini batch vs Lloyds: " << RSS_mini_batch_Lloyds[n][e] << endl;
		
				RSS_coresets_Lloyds[n][e] += 100. * (RSS_coresets_aux - RSS_standard_aux) / RSS_standard_aux / n_iter_N;
		//	    	cout << "RSS coresets vs Lloyds: " << RSS_coresets_Lloyds[n][e] << endl; 
		
				RSS_EKMeans_coresets[n][e] += 100. * (RSS_EKMeans_aux - RSS_coresets_aux) / RSS_coresets_aux / n_iter_N;
		//	    	cout << "RSS EKMeans vs coresets: " << RSS_EKMeans_coresets[n][e] << endl;   			    				    	
			    	
			    	long double Nij_EKMeans = 0, Nij_mini_batch = 0, Nij_coresets = 0, Nij_EKMeans_coresets = 0, ai = 0, bj_EKMeans = 0, bj_mini_batch = 0, bj_coresets = 0;
			    	long double MI_EKMeans = 0, MI_mini_batch = 0, MI_coresets = 0, MI_EKMeans_coresets = 0, H_standard = 0, H_EKMeans = 0, H_mini_batch = 0, H_coresets = 0;
			    	for (int i = 0; i < K; i++) {
			    		for (int j = 0; j < K; j++) {
			    		
			    		     	Nij_EKMeans += overlap_matrix_EKMeans_Lloyds[i][j] * (overlap_matrix_EKMeans_Lloyds[i][j] - 1)/2;
			    		     	Nij_mini_batch += overlap_matrix_mini_batch_Lloyds[i][j] * (overlap_matrix_mini_batch_Lloyds[i][j] - 1)/2;
			    		     	Nij_coresets += overlap_matrix_coresets_Lloyds[i][j] * (overlap_matrix_coresets_Lloyds[i][j] - 1)/2;
			    		     	Nij_EKMeans_coresets += overlap_matrix_EKMeans_coresets[i][j] * (overlap_matrix_EKMeans_coresets[i][j] - 1)/2;
			    		     	
			    		     	if (overlap_matrix_EKMeans_Lloyds[i][j] > 0)
			    		     		MI_EKMeans += overlap_matrix_EKMeans_Lloyds[i][j] * log( N[n] * overlap_matrix_EKMeans_Lloyds[i][j] / marginal_distribution_standard[i] / marginal_distribution_EKMeans[j]);     	
			    		     	if (overlap_matrix_mini_batch_Lloyds[i][j] > 0)
			    		     		MI_mini_batch += overlap_matrix_mini_batch_Lloyds[i][j] * log( N[n] * overlap_matrix_mini_batch_Lloyds[i][j] / marginal_distribution_standard[i] / marginal_distribution_mini_batch[j]);
			    		     	if (overlap_matrix_coresets_Lloyds[i][j] > 0)
			    		     		MI_coresets += overlap_matrix_coresets_Lloyds[i][j] * log( N[n] * overlap_matrix_coresets_Lloyds[i][j] / marginal_distribution_standard[i] / marginal_distribution_coresets[j]);
			    		     	if (overlap_matrix_EKMeans_coresets[i][j] > 0)
			    		     		MI_EKMeans_coresets += overlap_matrix_EKMeans_coresets[i][j] * log( N[n] * overlap_matrix_EKMeans_coresets[i][j] / marginal_distribution_coresets[i] / marginal_distribution_EKMeans[j]);
			    		     		
			    		}
			    		
			    		ai += marginal_distribution_standard[i] * (marginal_distribution_standard[i] - 1)/2;
			    		bj_EKMeans += marginal_distribution_EKMeans[i] * (marginal_distribution_EKMeans[i] - 1)/2;
			    		bj_mini_batch += marginal_distribution_mini_batch[i] * (marginal_distribution_mini_batch[i] - 1)/2;
			    		bj_coresets += marginal_distribution_coresets[i] * (marginal_distribution_coresets[i] - 1)/2;
			    		
			    		if (marginal_distribution_standard[i] > 0)
			    			H_standard += marginal_distribution_standard[i] * log( N[n] / marginal_distribution_standard[i]);
			    		if (marginal_distribution_EKMeans[i] > 0)
			    			H_EKMeans += marginal_distribution_EKMeans[i] * log( N[n] / marginal_distribution_EKMeans[i]);
			    		if (marginal_distribution_mini_batch[i] > 0)
			    			H_mini_batch += marginal_distribution_mini_batch[i] * log( N[n] / marginal_distribution_mini_batch[i]);
			    		if (marginal_distribution_coresets[i] > 0)
			    			H_coresets += marginal_distribution_coresets[i] * log( N[n] / marginal_distribution_coresets[i]);
			    	}					   	    	
			    		     
			    	ARI_EKMeans_Lloyds[n][e] +=  (Nij_EKMeans - 2 * ai * bj_EKMeans / N[n] / (N[n]-1) ) / (ai/2 + bj_EKMeans/2 - 2 * ai * bj_EKMeans / N[n] / (N[n]-1) ) / n_iter_N;
			    	ARI_mini_batch_Lloyds[n][e] += (Nij_mini_batch - 2 * ai * bj_mini_batch / N[n] / (N[n]-1) ) / (ai/2 + bj_mini_batch/2 - 2 * ai * bj_mini_batch / N[n] / (N[n]-1) ) / n_iter_N;
			    	ARI_coresets_Lloyds[n][e] += (Nij_coresets - 2 * ai * bj_coresets / N[n] / (N[n]-1) ) / (ai/2 + bj_coresets/2 - 2 * ai * bj_coresets / N[n] / (N[n]-1) ) / n_iter_N;
			    	ARI_EKMeans_coresets[n][e] += (Nij_EKMeans_coresets - 2 * bj_EKMeans * bj_coresets / N[n] / (N[n]-1) ) / (bj_EKMeans/2 + bj_coresets/2 - 2 * bj_EKMeans * bj_coresets / N[n] / (N[n]-1) ) / n_iter_N;
			    	NMI_EKMeans_Lloyds[n][e] += 2 * MI_EKMeans / (H_standard + H_EKMeans) / n_iter_N;				   				    	
			    	NMI_mini_batch_Lloyds[n][e] += 2 * MI_mini_batch / (H_standard + H_mini_batch) / n_iter_N;
			    	NMI_coresets_Lloyds[n][e] += 2 * MI_coresets / (H_standard + H_coresets) / n_iter_N;
			    	NMI_EKMeans_coresets[n][e] += 2 * MI_EKMeans_coresets / (H_coresets + H_EKMeans) / n_iter_N;
			    	
		//	    	cout << "ARI EKmeans: " << ARI_EKMeans_Lloyds[n][e] << endl;
		//	    	cout << "ARI mini batch: " << ARI_mini_batch_Lloyds[n][e] << endl;
		//	    	cout << "ARI coresets: " << ARI_coresets_Lloyds[n][e] << endl;
		//	    	cout << "NMI EKmeans: " << NMI_EKMeans_Lloyds[n][e] << endl;				 				 
		//	    	cout << "NMI mini batch: " << NMI_mini_batch_Lloyds[n][e] << endl;
		//	    	cout << "NMI coresets: " << NMI_coresets_Lloyds[n][e] << endl;
			}
			cout << n << ' ' << iter_N << endl; 						
		}
    	}
    	
    	// Write the output into a separate file
	ofstream MyFile;
	
	MyFile.open("EKmeans_vs_Lloyds_13.txt");
	MyFile << "n  eps  time  iterations  RSS  ARI  NMI" << endl;
	MyFile << "samples" << ' ' << n_iter_N << endl;
	
	for (int n = 0; n < length_N; n++) {
		MyFile << N[n] << ' ' << 0.0 << ' ' << time_standard_Kmeans[n] / ((double) n_iter_N) << ' ' << iterations_standard_Kmeans[n] / ((double) n_iter_N) << endl;
		cout << "N: " << N[n] << endl;
		cout << "Milliseconds standard: " << time_standard_Kmeans[n] / ((double) n_iter_N) << endl;
    		cout << "Iterations standard: " << iterations_standard_Kmeans[n] / ((double) n_iter_N) << endl << endl;
    		
		for (int e = 0; e < length_eps; e++) {
			MyFile << N[n] << ' ' << eps[e] << ' ' << time_EKMeans[n][e] / ((double) n_iter_N) << ' ' << iterations_EKMeans[n][e] / ((double) n_iter_N) << ' ' << RSS_EKMeans_Lloyds[n][e] << ' ' << ARI_EKMeans_Lloyds[n][e] << ' ' << NMI_EKMeans_Lloyds[n][e] << endl;
			cout << "Miliseconds EKmeans: " << time_EKMeans[n][e] / ((double) n_iter_N) << endl;
    			cout << "Iterations EKmeans: " << iterations_EKMeans[n][e] / ((double) n_iter_N) << endl;
    			cout << "RSS EKmeans vs Lloyds (%): " << RSS_EKMeans_Lloyds[n][e] << endl;
    			cout << "ARI EKmeans vs Lloyds: " << ARI_EKMeans_Lloyds[n][e] << endl;
    			cout << "NMI EKmeans vs Lloyds: " << NMI_EKMeans_Lloyds[n][e] << endl << endl;
    		}
	}
	MyFile.close();	
	
	MyFile.open("mini_batch_vs_Lloyds_13.txt");
	MyFile << "n  eps  time  iterations  RSS  ARI  NMI" << endl;
	MyFile << "samples" << ' ' << n_iter_N << endl;
		
	for (int n = 0; n < length_N; n++) {
		MyFile << N[n] << ' ' << 0.0 << ' ' << time_standard_Kmeans[n] / ((double) n_iter_N) << ' ' << iterations_standard_Kmeans[n] / ((double) n_iter_N) << endl;
		cout << "N: " << N[n] << endl;
		cout << "Milliseconds standard: " << time_standard_Kmeans[n] / ((double) n_iter_N) << endl;
    		cout << "Iterations standard: " << iterations_standard_Kmeans[n] / ((double) n_iter_N) << endl << endl;
    		
		for (int e = 0; e < length_eps; e++) {
			MyFile << N[n] << ' ' << eps[e] << ' ' << time_mini_batch[n][e] / ((double) n_iter_N) << ' ' << iterations_mini_batch[n][e] / ((double) n_iter_N) << ' ' << RSS_mini_batch_Lloyds[n][e] << ' ' << ARI_mini_batch_Lloyds[n][e] << ' ' << NMI_mini_batch_Lloyds[n][e] << endl;
			cout << "Miliseconds mini batch: " << time_mini_batch[n][e] / ((double) n_iter_N) << endl;
    			cout << "Iterations mini batch: " << iterations_mini_batch[n][e] / ((double) n_iter_N) << endl;
    			cout << "RSS mini batch vs Lloyds (%): " << RSS_mini_batch_Lloyds[n][e] << endl;
    			cout << "ARI mini batch vs Lloyds: " << ARI_mini_batch_Lloyds[n][e] << endl;
    			cout << "NMI mini batch vs Lloyds: " << NMI_mini_batch_Lloyds[n][e] << endl << endl;
    		}
	}
	MyFile.close();
	
	MyFile.open("coresets_vs_Lloyds_13.txt");
	MyFile << "n  eps  time  iterations  RSS  ARI  NMI" << endl;
	MyFile << "samples" << ' ' << n_iter_N << endl;
	
	for (int n = 0; n < length_N; n++) {
		MyFile << N[n] << ' ' << 0.0 << ' ' << time_standard_Kmeans[n] / ((double) n_iter_N) << ' ' << iterations_standard_Kmeans[n] / ((double) n_iter_N) << endl;
		cout << "N: " << N[n] << endl;
		cout << "Milliseconds standard: " << time_standard_Kmeans[n] / ((double) n_iter_N) << endl;
    		cout << "Iterations standard: " << iterations_standard_Kmeans[n] / ((double) n_iter_N) << endl << endl;
    		
		for (int e = 0; e < length_eps; e++) {
			MyFile << N[n] << ' ' << eps[e] << ' ' << time_coresets[n][e] / ((double) n_iter_N) << ' ' << iterations_coresets[n][e] / ((double) n_iter_N) << ' ' << RSS_coresets_Lloyds[n][e] << ' ' << ARI_coresets_Lloyds[n][e] << ' ' << NMI_coresets_Lloyds[n][e] << endl;
			cout << "Miliseconds coresets: " << time_coresets[n][e] / ((double) n_iter_N) << endl;
    			cout << "Iterations coresets: " << iterations_coresets[n][e] / ((double) n_iter_N) << endl;
    			cout << "RSS coresets vs Lloyds (%): " << RSS_coresets_Lloyds[n][e] << endl;
    			cout << "ARI coresets vs Lloyds: " << ARI_coresets_Lloyds[n][e] << endl;
    			cout << "NMI coresets vs Lloyds: " << NMI_coresets_Lloyds[n][e] << endl << endl;
    		}
	}
	MyFile.close();
	
	MyFile.open("EKmeans_vs_coresets_13.txt");
	MyFile << "n  eps  time_EKMeans  time_coresets  iterations_EKMeans  iterations_coresets  RSS  ARI  NMI" << endl;
	MyFile << "samples" << ' ' << n_iter_N << endl;
	
	for (int n = 0; n < length_N; n++) {
		for (int e = 0; e < length_eps; e++) {
			MyFile << N[n] << ' ' << eps[e] << ' ' << time_EKMeans[n][e] / ((double) n_iter_N) << ' ' << time_coresets[n][e] / ((double) n_iter_N) << ' ' << iterations_EKMeans[n][e] / ((double) n_iter_N) << ' ' << iterations_coresets[n][e] / ((double) n_iter_N) << ' ' << RSS_EKMeans_coresets[n][e] << ' ' << ARI_EKMeans_coresets[n][e] << ' ' << NMI_EKMeans_coresets[n][e] << endl;
			cout << "Miliseconds EKmeans: " << time_EKMeans[n][e] / ((double) n_iter_N) << endl;
			cout << "Miliseconds coresets: " << time_coresets[n][e] / ((double) n_iter_N) << endl;
    			cout << "Iterations EKmeans: " << iterations_EKMeans[n][e] / ((double) n_iter_N) << endl;
    			cout << "Iterations coresets: " << iterations_coresets[n][e] / ((double) n_iter_N) << endl;
    			cout << "RSS EKmeans vs coresets (%): " << RSS_EKMeans_coresets[n][e] << endl;
    			cout << "ARI EKmeans vs coresets: " << ARI_EKMeans_coresets[n][e] << endl;
    			cout << "NMI EKmeans vs coresets: " << NMI_EKMeans_coresets[n][e] << endl << endl;
    		}
	}
	MyFile.close();
	
	MyFile.open("preprocessing_EK_means_coresets_13.txt");
	MyFile << "n  time_EKMeans  time_coresets" << endl;
	MyFile << "samples" << ' ' << n_iter_N << endl;
	
	for (int n = 0; n < length_N; n++) {
		cout << "N: " << N[n] << endl;
		MyFile << N[n] << ' ' << time_EKMeans_preprocessing[n] / ((double) n_iter_N) << ' ' << time_coresets_preprocessing[n] / ((double) n_iter_N) << endl;		
		cout << "Miliseconds EKMeans preprocessing: " << time_EKMeans_preprocessing[n] / ((double) n_iter_N) << endl;
		cout << "Miliseconds coresets preprocessing: " << time_coresets_preprocessing[n] / ((double) n_iter_N) << endl;
	}
	MyFile.close();
	
	auto t1 = chrono::high_resolution_clock::now();
	cout <<  chrono::duration_cast<chrono::seconds>(t1 - t0).count() << endl;
    	
    	return 0;

}
