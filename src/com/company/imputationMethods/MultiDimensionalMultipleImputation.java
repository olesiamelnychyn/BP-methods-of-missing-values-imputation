package com.company.imputationMethods;

import com.company.utils.objects.MainData;
import com.company.utils.objects.Statistics;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

import java.util.ArrayList;
import java.util.stream.Collectors;

import static com.company.utils.ColorFormatPrint.ANSI_PURPLE_BACKGROUND;
import static com.company.utils.ColorFormatPrint.ANSI_RESET;

public class MultiDimensionalMultipleImputation extends ImputationMethod {
	MainData data;
	SimpleDataSet datasetMissing;
	Statistics stat;
	ArrayList<ImputationMethod> methods = new ArrayList<>();


	public MultiDimensionalMultipleImputation (MainData data, SimpleDataSet datasetMissing, Statistics stat) {
		super(data);
		this.data = data;
		this.datasetMissing = datasetMissing;
		this.stat = stat;
	}

	public void preprocessData () {
		// no preprocessing needed, it is done inside methods
	}

	public void fit () {
		// select best methods for each predictor separately
		SimpleImputationMethods simpleImputationMethods = new SimpleImputationMethods(datasetMissing);
		for (int columnPredictor : data.getColumnPredictors()) {
			MainData mainData = new MainData(new int[]{columnPredictor}, columnPredicted, data.getDp(), data.getTrain(), data.getImpute(), data.isMultiple());
			methods.add(simpleImputationMethods.imputeSimple(mainData, stat));
		}
	}

	public double predict (DataPoint dp) {
		// get all the values predicted by the methods
		return methods.stream()
			.map(method -> {
				method.preprocessData();
				method.fit();
				return method.predict(dp);
			})
			//return average of all predictors
			.reduce(0.0, Double::sum) / methods.size();
	}

	public void print () {
		System.out.println(ANSI_PURPLE_BACKGROUND + "MultiDimensional" + ANSI_RESET);
		System.out.println("Methods: " + methods.stream().map(ImputationMethod::getClass).map(Class::getName).collect(Collectors.joining(", ")));
	}
}