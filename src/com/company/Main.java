package com.company;

import com.company.utils.DatasetManipulation;
import com.company.utils.ImputationMethods;
import jsat.SimpleDataSet;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class Main {

	public static void main (String[] args) throws IOException {

		SimpleDataSet dataset = DatasetManipulation.readDataset("src/com/company/data/combined_csv_new.csv");
		int[] columnPredictors = new int[dataset.getDataMatrix().cols() - 1];
		int columnPredicted = 3;
		int k = 0;
		for (int i = 0; i < dataset.getDataMatrix().cols() - 1; i++) {
			if (k != columnPredicted) {
				columnPredictors[i] = k++;
			} else {
				columnPredictors[i] = ++k;
				k++;
			}
		}

		ImputationMethods imputationMethods = new ImputationMethods(columnPredicted, dataset);

		String str = "Results:\n\n";
		BufferedWriter writer = new BufferedWriter(new FileWriter("src/com/company/results.txt"));
		writer.write(str);
		writer.close();

		//Linear Regression
		for (int columnPredictor : columnPredictors) {
			imputationMethods.LinearRegressionJSAT(columnPredictor);
		}

		//Multiple Linear Regression from JSAT
		imputationMethods.MultipleLinearRegressionJSAT();

		//Polynomial Curve Fitter
		for (int columnPredictor : columnPredictors) {
			imputationMethods.PolynomialCurveFitterApache(columnPredictor);
		}

//		Gaussian Curve Fitter - takes some time
		for (int columnPredictor : columnPredictors) {
			imputationMethods.GaussianCurveFitterApache(columnPredictor);
		}

//		Linear Interpolator - values must be strictly increasing
//		for(int columnPredictor : columnPredictors){
//			imputationMethods.LinearInterpolatorApache(columnPredictor);
//		}

		//Polynomial Regression from JAMA
		for (int columnPredictor : columnPredictors) {
			imputationMethods.PolynomialRegressionJama(columnPredictor);
		}

		//Multiple Linear Regression from JAMA
		imputationMethods.MultipleLinearRegressionJama(columnPredictors);

		//Multiple Polynomial Regression from JAMA
		imputationMethods.MultiplePolynomialRegressionJama(columnPredictors, 2);
	}

}
