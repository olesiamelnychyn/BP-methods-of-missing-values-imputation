package com.company;

import com.company.utils.DatasetManipulation;
import com.company.utils.Regressions;
import jsat.SimpleDataSet;

import java.io.IOException;

public class Main {

	public static void main (String[] args) throws IOException {

		SimpleDataSet dataset = DatasetManipulation.readDataset("src/com/company/data/Dataset.csv");
		int columnPredicted = 4;
		SimpleDataSet original = DatasetManipulation.createDeepCopy(dataset, 10, 20);
		Regressions regressions = new Regressions(columnPredicted, dataset, original);

		//Linear Regression
		for (int columnPredictor = 0; columnPredictor < 4; columnPredictor++) {
			regressions.LinearRegression(columnPredictor);
		}

		//Multiple Linear Regression
		regressions.MultipleLinearRegression();
	}
}
