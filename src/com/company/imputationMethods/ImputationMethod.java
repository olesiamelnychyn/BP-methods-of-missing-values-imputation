package com.company.imputationMethods;

import com.sun.tools.javac.util.List;
import jsat.SimpleDataSet;
import jsat.classifiers.DataPoint;

import java.util.ArrayList;

public abstract class ImputationMethod {
	protected SimpleDataSet trainingCopy;
	protected SimpleDataSet toBePredicted;
	protected int columnPredicted;

	protected ImputationMethod (int columnPredicted, ArrayList<SimpleDataSet> datasets) {
		this.columnPredicted = columnPredicted;
		trainingCopy = datasets.get(0);
		toBePredicted = datasets.get(1);
	}

	public abstract void preprocessData ();

	public abstract void fit ();

	public abstract double predict (DataPoint dp);

	public abstract void print ();

	public SimpleDataSet getToBePredicted () {
		return toBePredicted;
	}

	public SimpleDataSet getTrainingCopy () {
		return trainingCopy;
	}

	public int getColumnPredicted () {
		return columnPredicted;
	}

	public ArrayList<SimpleDataSet> getDatasets () {
		return new ArrayList<>(List.of(trainingCopy, toBePredicted));
	}
}
