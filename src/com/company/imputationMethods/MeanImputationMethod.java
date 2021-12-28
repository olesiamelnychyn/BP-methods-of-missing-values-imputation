package com.company.imputationMethods;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

import java.util.ArrayList;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;

public class MeanImputationMethod extends ImputationMethod {
	private double mean;

	public MeanImputationMethod (int columnPredicted, ArrayList<SimpleDataSet> datasets) {
		super(columnPredicted, datasets);
	}

	public void preprocessData () {
	}

	public void fit () {
		mean = trainingCopy.getDataMatrix().getColumn(columnPredicted).mean();
	}

	public double predict (DataPoint dp) {
		return mean;
	}

	public void print () {
		System.out.println(ANSI_PURPLE_BACKGROUND + "Mean Imputation" + ANSI_RESET + "\nMean: [" + mean + "]");
	}
}
