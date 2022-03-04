package com.company.imputationMethods;

import com.company.utils.objects.MainData;
import jsat.classifiers.DataPoint;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;

public class MedianImputationMethod extends ImputationMethod {
	private double median;

	public MedianImputationMethod (MainData data) {
		super(data);
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
