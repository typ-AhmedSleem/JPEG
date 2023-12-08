from numpy import mean, power, float32


class Metric:
    def calculate(source, final):
        """Base method that should be overridden in a certain metric

        Args:
            source: The source of the image before being encoded.
            final: The final result after encoding completed.
        """
        pass


class CRMetric(Metric):
    """Compression Ratio metric known as (CR) that calculates its metric value
    by dividing the source value by the final value.
    """

    def calculate(self, source, final):
        """Calculates the metric value by dividing the source value by\n
        the final value and return its result.

        Args:
            source (int): Size of image before compression.
            final (int): Size of image after compression.

        Returns:
            float: Compression Ratio metric result.
        """
        return source / final


class MSEMetric(Metric):
    """Mean Square Error metric known as (MSE) that
    [NOTE]: Return type of this
    """

    def calculate(self, source, final):
        """Calculates the metric value as follows:\n
        1-Convert the dtype of both 'source' and 'final' to float32.\n
        2-Subtract 'final' from 'source'.\n
        3-Raise the subtraction result to the power of 2.\n
        4-.Calculate the mean of result from step above.\n

        Args:
            source (cv2.typing.MatLike): Source 3-channels image as a matrix.
            final (cv2.typing.MatLike): Compressed 3-channels image as a matrix.

        Returns:
            float: MSE metric result.
        """
        POWER = 2
        return mean(power((source.astype(float32) - final.astype(float32)), POWER))
