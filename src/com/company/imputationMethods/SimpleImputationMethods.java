package com.company.imputationMethods;

import com.company.utils.DatasetManipulation;
import com.company.utils.objects.MainData;
import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;

import static com.company.utils.calculations.StatCalculations.*;

public class SimpleImputationMethods {
	SimpleDataSet datasetMissing;

	public SimpleImputationMethods (SimpleDataSet datasetMissing) {
		this.datasetMissing = datasetMissing;
	}

	public ImputationMethod imputeSimple (MainData data, Statistics stat) {
		if (data.getTrain() == null) {
			// prepare datasets
			DatasetManipulation.getToBeImputedAndTrainDeepCopiesAroundIndex(data, datasetMissing, datasetMissing.getDataPoints().indexOf(data.getDp()), 8);
		}

		// select appropriate imputation method
		if (isCloseToMean(data, stat)) {
			return new MeanImputationMethod(data);
		} else if (isCloseToMedian(data, stat)) {
			return new MedianImputationMethod(data);
		} else if (data.getColumnPredictors().length == 1) {
			int columnPredictor = data.getColumnPredictors()[0];
			if (isStrictlyIncreasing(data, data.getColumnPredicted()) && isStrictlyIncreasing(data, columnPredictor)) {
				return new LinearInterpolatorApacheMethod(data, true);
			} else if (isStrictlyDecreasing(data, data.getColumnPredicted()) && isStrictlyDecreasing(data, columnPredictor)) {
				return new LinearInterpolatorApacheMethod(data, false);
			} else if (hasLinearRelationship(data, stat)) {
				return new LinearRegressionJSATMethod(data);
			} else {
				int order = getPolynomialOrderSimple(data, stat);
				if (order != -1) {
					return new PolynomialCurveFitterApacheMethod(data, order);
				}
			}
		}
		return new MeanImputationMethod(data);
	}
}
