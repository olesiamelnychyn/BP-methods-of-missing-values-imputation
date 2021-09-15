package com.company;

import com.company.utils.DatasetManipulation;
import com.company.utils.ImputationMethods;
import jsat.SimpleDataSet;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class Main {

	public static void main (String[] args) throws IOException {

		SimpleDataSet dataset = DatasetManipulation.readDataset("src/com/company/data/Dataset.csv");
		int columnPredicted = 4;
		ImputationMethods imputationMethods = new ImputationMethods(columnPredicted, dataset);

		String str = "Results:\n\n";
		BufferedWriter writer = new BufferedWriter(new FileWriter("src/com/company/results.txt"));
		writer.write(str);
		writer.close();

		//Linear Regression
		for (int columnPredictor = 0; columnPredictor < 4; columnPredictor++) {
			imputationMethods.LinearRegressionJSAT(columnPredictor);
		}

		//Multiple Linear Regression from JSAT
		imputationMethods.MultipleLinearRegressionJSAT();

		//Polynomial Curve Fitter
		for (int columnPredictor = 0; columnPredictor < 4; columnPredictor++) {
			imputationMethods.PolynomialCurveFitterApache(columnPredictor);
		}

//		Gaussian Curve Fitter - takes some time
		for (int columnPredictor = 0; columnPredictor < 4; columnPredictor++) {
			imputationMethods.GaussianCurveFitterApache(columnPredictor);
		}

//		Linear Interpolator - values must be strictly increasing
//		for (int columnPredictor = 0; columnPredictor < 4; columnPredictor++) {
//			imputationMethods.LinearInterpolatorApache(columnPredictor);
//		}

		//Polynomial Regression from JAMA
		for (int columnPredictor = 0; columnPredictor < 4; columnPredictor++) {
			imputationMethods.PolynomialRegressionJama(columnPredictor);
		}

		//Multiple Linear Regression from JAMA
		imputationMethods.MultipleLinearRegressionJama();
	}

}
