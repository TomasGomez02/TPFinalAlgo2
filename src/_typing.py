import numpy as np
import pandas as pd
import numpy.typing
from scipy.sparse import spmatrix

DecisionResult = str | float
ArrayLike = numpy.typing.ArrayLike
MatrixLike = np.ndarray | pd.DataFrame | spmatrix