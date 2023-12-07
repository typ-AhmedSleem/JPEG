from numpy import mean, power, float32

class Metric:
    
    def calculate(source, final):
        pass

class CRMetric(Metric):
    
    def calculate(self, source, final):
        return source / final
    
class MSEMetric(Metric):
    
    def calculate(self, source, final):
        POWER = 2
        return mean(power((source.astype(float32) - final.astype(float32)), POWER))
