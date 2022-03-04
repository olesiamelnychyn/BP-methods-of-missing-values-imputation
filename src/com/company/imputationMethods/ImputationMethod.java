package com.company.imputationMethods;

import com.company.utils.objects.MainData;
import com.sun.tools.javac.util.List;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

import java.util.ArrayList;

public abstract class ImputationMethod {
	protected SimpleDataSet trainingCopy;
	protected SimpleDataSet toBePredicted;
	protected int columnPredicted;
	protected MainData data;

	protected ImputationMethod (MainData data) {
		this.data = data;
		this.columnPredicted = data.getColumnPredicted();
		trainingCopy = data.getTrain();
		toBePredicted = data.getImpute();
	}

	public abstract void preprocessData ();

	public abstract void fit ();

	public abstract double predict (DataPoint dp);

	public abstract void print ();

	public SimpleDataSet getToBePredicted () {
		return toBePredicted;
	}

	public int getColumnPredicted () {
		return columnPredicted;
	}

	public MainData getData () {
		return data;
	}

	public ArrayList<SimpleDataSet> getDatasets () {
		return new ArrayList<>(List.of(trainingCopy, toBePredicted));
	}
}
