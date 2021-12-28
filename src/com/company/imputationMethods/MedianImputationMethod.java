package com.company.imputationMethods;

import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

import java.util.ArrayList;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;

public class MedianImputationMethod extends ImputationMethod {
	private double median;

	public MedianImputationMethod (int columnPredicted, ArrayList<SimpleDataSet> datasets) {
		super(columnPredicted, datasets);
	}

	public void preprocessData () {
	}

	public void fit () {
		median = trainingCopy.getDataMatrix().getColumn(columnPredicted).median();
	}

	public double predict (DataPoint dp) {
		return median;
	}

	public void print () {
		System.out.println(ANSI_PURPLE_BACKGROUND + "Median Imputation" + ANSI_RESET + "\nMedian: [" + median + "]");
	}
}
