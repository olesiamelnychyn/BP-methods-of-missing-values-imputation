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
		int columnPredictor = data.getColumnPredictors()[0];

		// select appropriate imputation method
		if (isCloseToMean(data.getTrain(), stat)) {
			return new MeanImputationMethod(data);
		} else if (isCloseToMedian(data.getTrain(), stat)) {
			return new MedianImputationMethod(data);
		} else if (isStrictlyIncreasing(data.getTrain(), data.getColumnPredicted()) && isStrictlyIncreasing(data.getTrain(), columnPredictor)) {
			return new LinearInterpolatorApacheMethod(data, true);
		} else if (isStrictlyDecreasing(data.getTrain(), data.getColumnPredicted()) && isStrictlyDecreasing(data.getTrain(), columnPredictor)) {
			return new LinearInterpolatorApacheMethod(data, false);
		} else if (hasLinearRelationship(data.getTrain(), stat)) {
			return new LinearRegressionJSATMethod(data);
		} else {
			int order = getPolynomialOrder(data.getTrain(), stat);
			if (order != -1) {
				return new PolynomialCurveFitterApacheMethod(data, order);
			}
		}
		return new MeanImputationMethod(data);
	}
}
