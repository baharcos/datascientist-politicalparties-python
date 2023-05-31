from pathlib import Path

from matplotlib import pyplot

from scipy.stats import multivariate_normal
import numpy as np

from political_party_analysis.loader import DataLoader
from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.estimator import DensityEstimator
from political_party_analysis.visualization import scatter_plot
from political_party_analysis.visualization import plot_density_estimation_results

if __name__ == "__main__":

    data_loader = DataLoader()
    # Data pre-processing step
    data_loader.preprocess_data()

    # Dimensionality reduction step
    dimension_reducer = DimensionalityReducer(data_loader.party_data)
    reduced_dim_data = dimension_reducer.transform_data()

    ## Uncomment this snippet to plot dim reduced data
    pyplot.figure()
    splot = pyplot.subplot()
    scatter_plot(
        reduced_dim_data,
        color="r",
        splot=splot,
        label="dim reduced data",
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "dim_reduced_data.png"]))

    # Density estimation/distribution modelling step
    density_estimator = DensityEstimator(data_loader.party_data, dimension_reducer, data_loader.party_data.columns)
    density_estimator.multivariate_normal_mle_estimator()

    # Plot density estimation results here
    plot_density_estimation_results()

    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "density_estimation.png"]))

    # Plot left and right wing parties here
    pyplot.figure()
    splot = pyplot.subplot()
    ##### YOUR CODE GOES HERE #####
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "left_right_parties.png"]))
    pyplot.title("Lefty/righty parties")

    # Plot finnish parties here
    ##### YOUR CODE GOES HERE #####

    print("Analysis Complete")
