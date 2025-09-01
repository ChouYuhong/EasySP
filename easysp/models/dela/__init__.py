# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from easysp.models.dela.configuration_dela import DELAConfig
from easysp.models.dela.modeling_dela import DELAForCausalLM, DELAModel

AutoConfig.register(DELAConfig.model_type, DELAConfig, exist_ok=True)
AutoModel.register(DELAConfig, DELAModel, exist_ok=True)
AutoModelForCausalLM.register(DELAConfig, DELAForCausalLM, exist_ok=True)


__all__ = ['DELAConfig', 'DELAForCausalLM', 'DELAModel']
