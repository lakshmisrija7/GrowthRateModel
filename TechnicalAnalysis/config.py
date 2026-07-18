WS_URL = "wss://asense.assetsense.com/wsasense"
API_KEY = "2FDE23B4B72FF51D74A4402227E61AD246057783B364A7D527EEB16A5D41F45C"
DEFAULT_FROM_DATE = None
DEFAULT_TO_DATE = None

class RSIConfig:
    class Zones:
        NEUTRAL = 1
        OVERBOUGHT = 2
        OVERSOLD = 3
    
    class Signals:
        NONE = 1
        BULLISH = 2
        BEARISH = 3

class ADXConfig:
    class States:
        WEAK = 1
        MEDIUM = 2
        STRONG = 3
    
    class Alerts:
        NONE = 1
        BULLISH = 2
        BEARISH = 3
        EXHAUSTION = 4

class BollingerConfig:
    class Breakouts:
        NONE = 1
        BULLISH = 2
        BEARISH = 3

class MACDConfig:
    class Momentum:
        GAINING = 1
        LOSING = 2
    
    class Divergence:
        NONE = 1
        BULLISH = 2
        BEARISH = 3


class ScorerConfig:
    class Trend:
        STRONG_BULLISH = 3
        BULLISH = 2
        WEAK_BULLISH = 1
        NEUTRAL = 0
        WEAK_BEARISH = -1
        BEARISH = -2
        STRONG_BEARISH = -3

    class Momentum:
        STRONG_BULLISH = 3
        BULLISH = 2
        WEAK_BULLISH = 1
        NEUTRAL = 0
        WEAK_BEARISH = -1
        BEARISH = -2
        STRONG_BEARISH = -3

    class Volatility:
        EXTREME_HIGH = 2
        HIGH = 1
        NORMAL = 0
        LOW = -1
        VERY_LOW = -2

    class Risk:
        BREAKOUT = 2
        BREAKDOWN = -2
        NEAR_RESISTANCE = -1
        NEAR_SUPPORT = 1
        NEUTRAL = 0
