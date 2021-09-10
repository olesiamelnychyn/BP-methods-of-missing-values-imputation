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
		SimpleDataSet original = DatasetManipulation.createDeepCopy(dataset, 10, 20);
		ImputationMethods imputationMethods = new ImputationMethods(columnPredicted, dataset, original);
		imputationMethods.setDatasetSize(14000);

		String str = "Results:\n\n";
		BufferedWriter writer = new BufferedWriter(new FileWriter("src/com/company/results.txt"));
		writer.write(str);
		writer.close();

		//Linear Regression
		for (int columnPredictor = 0; columnPredictor < 4; columnPredictor++) {
			imputationMethods.LinearRegression(columnPredictor);
		}

		//Multiple Linear Regression
		imputationMethods.MultipleLinearRegression();

		//Polynomial Curve Fitter
		for (int columnPredictor = 0; columnPredictor < 4; columnPredictor++) {
			imputationMethods.PolynomialCurveFitter(columnPredictor);
		}

		//Gaussian Curve Fitter - takes some time
//		for (int columnPredictor = 0; columnPredictor < 4; columnPredictor++) {
//			imputationMethods.GaussianCurveFitter(columnPredictor);
//		}

		//Linear Interpolator - values must be strictly increasing
//		for (int columnPredictor = 0; columnPredictor < 4; columnPredictor++) {
//			imputationMethods.LinearInterpolator(columnPredictor);
//		}
	}

}
