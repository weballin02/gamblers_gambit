# config.py
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class Config:
    YEARS: List[int] = (2023,)
    PBP_COLUMNS: List[str] = (
        "posteam",
        "defteam", 
        "play_type",
        "yards_gained",
        "epa"
    )
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_LEVEL: int = logging.INFO

# indicators.py
from typing import Optional
import pandas as pd
import nfl_data_py as nfl
import logging
from config import Config

logger = logging.getLogger(__name__)

class NFLDataManager:
    def __init__(self, config: Config):
        self.config = config
        self._data: Optional[pd.DataFrame] = None
    
    def load_pbp_data(self) -> pd.DataFrame:
        """Load and clean NFL play-by-play data."""
        try:
            logger.info(f"Loading PBP data for years: {self.config.YEARS}")
            pbp_data = nfl.import_pbp_data(
                years=self.config.YEARS,
                columns=self.config.PBP_COLUMNS
            )
            self._data = nfl.clean_nfl_data(pbp_data)
            logger.info("Successfully loaded PBP data")
            return self._data
        except Exception as e:
            logger.error(f"Error loading PBP data: {str(e)}")
            raise

    @property
    def data(self) -> pd.DataFrame:
        """Get cached data or load if not available."""
        if self._data is None:
            self._data = self.load_pbp_data()
        return self._data

class Indicators:
    def __init__(self, data_manager: NFLDataManager):
        self.data_manager = data_manager
    
    def avg_yards(self, team: str) -> float:
        """Calculate average yards for a team."""
        try:
            team_data = self.data_manager.data[
                self.data_manager.data['posteam'] == team
            ]
            return float(team_data['yards_gained'].mean()) if not team_data.empty else 0
        except Exception as e:
            logger.error(f"Error calculating avg_yards for {team}: {str(e)}")
            raise

    def avg_epa(self, team: str) -> float:
        """Calculate average EPA for a team."""
        try:
            team_data = self.data_manager.data[
                self.data_manager.data['posteam'] == team
            ]
            return float(team_data['epa'].mean()) if not team_data.empty else 0
        except Exception as e:
            logger.error(f"Error calculating avg_epa for {team}: {str(e)}")
            raise

# rules.py
from typing import Dict, List, Any, Callable
from dataclasses import dataclass

@dataclass
class Condition:
    indicator: str
    operator: str
    value: float

@dataclass
class Rule:
    team: str
    conditions: List[Condition]

class RuleEvaluator:
    def __init__(self, indicators: Indicators):
        self.indicators = indicators
        self.operator_map = {
            ">": lambda x, y: x > y,
            "<": lambda x, y: x < y,
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
            "==": lambda x, y: x == y
        }
        self.indicator_map = {
            "avg_yards": self.indicators.avg_yards,
            "avg_epa": self.indicators.avg_epa
        }

    def validate_rule(self, rule: Dict[str, Any]) -> Rule:
        """Validate and convert rule dict to Rule object."""
        try:
            conditions = [
                Condition(
                    indicator=c["indicator"],
                    operator=c["operator"],
                    value=float(c["value"])
                )
                for c in rule["conditions"]
            ]
            return Rule(team=rule["team"], conditions=conditions)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid rule format: {str(e)}")

    def evaluate_rule(self, rule: Dict[str, Any]) -> bool:
        """Evaluate a betting rule."""
        try:
            validated_rule = self.validate_rule(rule)
            results = []

            for condition in validated_rule.conditions:
                if condition.indicator not in self.indicator_map:
                    raise ValueError(f"Unknown indicator: {condition.indicator}")
                if condition.operator not in self.operator_map:
                    raise ValueError(f"Unknown operator: {condition.operator}")

                indicator_func = self.indicator_map[condition.indicator]
                operator_func = self.operator_map[condition.operator]
                
                actual_value = indicator_func(validated_rule.team)
                results.append(operator_func(actual_value, condition.value))

            return all(results)
        except Exception as e:
            logger.error(f"Error evaluating rule: {str(e)}")
            raise

# app.py
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import logging
from config import Config
from indicators import NFLDataManager, Indicators
from rules import RuleEvaluator

# Setup logging
logging.basicConfig(
    format=Config.LOG_FORMAT,
    level=Config.LOG_LEVEL
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize components
config = Config()
data_manager = NFLDataManager(config)
indicators = Indicators(data_manager)
rule_evaluator = RuleEvaluator(indicators)

@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler."""
    logger.error(f"Error: {str(error)}")
    if isinstance(error, BadRequest):
        return jsonify({"error": "Invalid request"}), 400
    return jsonify({"error": "Internal server error"}), 500

@app.route('/team-stats', methods=['GET'])
def get_team_stats():
    """Get statistics for a specific team."""
    team = request.args.get('team')
    if not team:
        raise BadRequest("Team parameter is required")
    
    try:
        return jsonify({
            "team": team,
            "avg_yards": indicators.avg_yards(team),
            "avg_epa": indicators.avg_epa(team)
        })
    except Exception as e:
        logger.error(f"Error getting team stats: {str(e)}")
        raise

@app.route('/evaluate-strategy', methods=['POST'])
def evaluate_strategy():
    """Evaluate a betting strategy."""
    if not request.is_json:
        raise BadRequest("Content-Type must be application/json")
    
    rule = request.json.get('rule')
    if not rule:
        raise BadRequest("Rule is required")
    
    try:
        result = rule_evaluator.evaluate_rule(rule)
        return jsonify({"bet": result})
    except Exception as e:
        logger.error(f"Error evaluating strategy: {str(e)}")
        raise

if __name__ == "__main__":
    app.run(debug=True)