from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Protocol, Tuple

from pydantic import ValidationError

from app.models.environment import (
    CoefficientsInput,
    EnvironmentalEstimatesRequest,
    EnvironmentalEstimatesResponse,
)


class LoggerProtocol(Protocol):
    def log(self, payload: Dict[str, object]) -> Tuple[bool, Optional[str]]:  # (logged, log_id)
        ...


@dataclass
class NoOpLogger:
    def log(self, payload: Dict[str, object]) -> Tuple[bool, Optional[str]]:
        return False, None


class EnvironmentalEstimationService:
    """Service providing deterministic environmental estimates.

    Formulas:
    - Volume with height: V = c * D^2 * H  [dm^3], default c=0.039
    - Volume without height: V = c * D^2   [dm^3], default c=0.77
    - Biomass (non-log): Y = a * D^b       [kg], default a=0.035, b=2.71
    - Biomass (log-form): ln(Y) = ln(a) + b*ln(D) => Y = exp(ln(a)+b*ln(D))
    - Carbon stock: carbon_stock = biomass * carbon_fraction [kg]
    - RSR (root-to-shoot ratio) used: override or default 0.25
    """

    _MODEL_VERSION = "plantaai-1.0.0"

    def __init__(self, logger: Optional[LoggerProtocol] = None) -> None:
        self._logger: LoggerProtocol = logger or NoOpLogger()

    def computeEnvironmentalEstimates(
        self, request: EnvironmentalEstimatesRequest
    ) -> EnvironmentalEstimatesResponse:
        try:
            # Pydantic validation already enforced on request
            normalized_inputs = self._normalize_inputs(request)

            # 2) Volume
            volume_dm3, volume_note = self._compute_volume_dm3(
                diameter_cm=normalized_inputs["tree"]["diameter_cm"],
                height_m=normalized_inputs["tree"].get("height_m"),
                coeffs=normalized_inputs["coeffs"],
            )

            # 3) Biomass
            biomass_kg = self._compute_biomass_kg(
                diameter_cm=normalized_inputs["tree"]["diameter_cm"],
                use_log_form=normalized_inputs["method"]["use_log_form"],
                coeffs=normalized_inputs["coeffs"],
            )

            # 4) Carbon stock
            carbon_stock_kg = biomass_kg * normalized_inputs["tree"]["carbon_fraction"]

            # 5) RSR used
            rsr_used = (
                normalized_inputs["method"]["rsr_override"]
                if normalized_inputs["method"]["rsr_override"] is not None
                else 0.25
            )

            # 6) BEF (optional)
            bef_value, bef_note = self._compute_bef(
                mode=normalized_inputs["method"]["bef_mode"],
                inputs=normalized_inputs,
            )

            # 7) Confidence & RD
            confidence_method = "analytical"
            confidence_notes = []
            if volume_note:
                confidence_notes.append(volume_note)
            if bef_note:
                confidence_notes.append(bef_note)

            rd_value: Optional[float] = None
            if normalized_inputs["feedback"]["observed_biomass_kg"] is not None:
                observed = normalized_inputs["feedback"]["observed_biomass_kg"]  # type: ignore[assignment]
                rd_value = abs(observed - biomass_kg) / observed if observed > 0 else None

            # 8) Logging (no-op safe)
            log_payload = {
                "request_id": normalized_inputs["meta"]["request_id"],
                "model_version": self._MODEL_VERSION,
                "inputs_normalized": normalized_inputs,
                "outputs": {
                    "volume_dm3": round(volume_dm3, 6),
                    "biomass_kg": round(biomass_kg, 6),
                    "carbon_stock_kg": round(carbon_stock_kg, 6),
                    "rsr_used": rsr_used,
                    "bef": None if bef_value is None else round(bef_value, 6),
                },
                "rd": None if rd_value is None else round(rd_value, 6),
                "timestamp": datetime.utcnow().isoformat(),
            }
            try:
                logged, log_id = self._logger.log(log_payload)
            except Exception:
                logged, log_id = False, None

            # 9) Response
            response = EnvironmentalEstimatesResponse(
                request_id=normalized_inputs["meta"]["request_id"],
                model_version=self._MODEL_VERSION,
                inputs=normalized_inputs,
                results={
                    "volume_dm3": round(volume_dm3, 6),
                    "biomass_kg": round(biomass_kg, 6),
                    "carbon_stock_kg": round(carbon_stock_kg, 6),
                    "rsr_used": rsr_used,
                    "bef": None if bef_value is None else round(bef_value, 6),
                    "confidence": {
                        "method": confidence_method,
                        "notes": "; ".join(confidence_notes) if confidence_notes else "",
                        "relative_error_rd": None if rd_value is None else round(rd_value, 6),
                    },
                },
                citations=[
                    {
                        "source": "Cutini et al., 2013",
                        "equations": [
                            "V=0.039*D^2*H",
                            "Y=a*D^b",
                            "ln(Y)=ln(a)+b*ln(D)",
                        ],
                    }
                ],
                logging={"logged": bool(logged), "log_id": log_id},
            )
            return response
        except ValidationError:
            # Propagate; FastAPI/request layer should control mapping to error responses
            raise
        except Exception as exc:
            raise exc

    def _normalize_inputs(self, request: EnvironmentalEstimatesRequest) -> Dict[str, object]:
        # Defaults
        carbon_fraction = request.tree.carbon_fraction if request.tree.carbon_fraction is not None else 0.47
        coeffs: CoefficientsInput = request.coeffs or CoefficientsInput()
        normalized = {
            "tree": {
                "diameter_cm": float(request.tree.diameter_cm),
                "height_m": None if request.tree.height_m is None else float(request.tree.height_m),
                "wood_density_kg_m3": None if request.tree.wood_density_kg_m3 is None else float(request.tree.wood_density_kg_m3),
                "carbon_fraction": float(carbon_fraction),
            },
            "site": {
                "site_id": request.site.site_id,
                "lat": request.site.lat,
                "lon": request.site.lon,
            },
            "method": {
                "use_log_form": bool(request.method.use_log_form),
                "rsr_override": request.method.rsr_override,
                "bef_mode": request.method.bef_mode,
            },
            "feedback": {
                "observed_biomass_kg": None if request.feedback is None else request.feedback.observed_biomass_kg,
                "notes": None if request.feedback is None else request.feedback.notes,
            },
            "meta": {
                "request_id": request.meta.request_id,
                "timestamp": request.meta.timestamp.isoformat(),
                "source": request.meta.source,
            },
            "coeffs": {
                "volume_with_h_coef": coeffs.volume_with_h_coef,
                "volume_without_h_coef": coeffs.volume_without_h_coef,
                "biomass_a": coeffs.biomass_a,
                "biomass_b": coeffs.biomass_b,
            },
        }
        return normalized

    def _compute_volume_dm3(
        self,
        diameter_cm: float,
        height_m: Optional[float],
        coeffs: Dict[str, object],
    ) -> Tuple[float, Optional[str]]:
        if height_m is not None and height_m > 0:
            volume = float(coeffs["volume_with_h_coef"]) * (diameter_cm ** 2) * height_m
            return volume, None
        volume = float(coeffs["volume_without_h_coef"]) * (diameter_cm ** 2)
        return volume, "Height missing; used D-only volume model"

    def _compute_biomass_kg(
        self,
        diameter_cm: float,
        use_log_form: bool,
        coeffs: Dict[str, object],
    ) -> float:
        a = float(coeffs["biomass_a"])  # default 0.035
        b = float(coeffs["biomass_b"])  # default 2.71
        if use_log_form:
            return math.exp(math.log(a) + b * math.log(diameter_cm))
        return a * (diameter_cm ** b)

    def _compute_bef(self, mode: str, inputs: Dict[str, object]) -> Tuple[Optional[float], Optional[str]]:
        if mode == "none":
            return None, None
        # In absence of additional inputs (e.g., stand-level or stem/volume specifics),
        # return null with an explanatory note.
        return None, "BEF not computed due to insufficient input for mode '" + mode + "'"


