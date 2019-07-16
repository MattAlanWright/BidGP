#ifndef _INSTRUCTION_HPP
#define _INSTRUCTION_HPP

// Effolkronium random library
#include "random.hpp"
using Random = effolkronium::random_static;

enum InstructionMode {
    INST_REGISTER_MODE,
    INST_INPUT_MODE,
    INST_NUM_MODES
};

enum InstructionOperation {
    INST_ADDITION,
    INST_SUBTRACTION,
    INST_MULTIPLICATION,
    INST_DIVISION,
    INST_COSINE,
    INST_LOG,
    INST_EXPONENTIAL,
    INST_CONDITIONAL,
    INST_NUM_OPERATIONS
};

enum InstructionComponent {
    INST_SOURCE,
    INST_OP,
    INST_TARGET,
    INST_MODE,
    INST_NUM_COMPONENTS
};

struct Instruction {

    int instruction[INST_NUM_COMPONENTS];
    int max_component_values[INST_NUM_COMPONENTS];

    Instruction(int num_features) : instruction { 0 }
    {
        max_component_values[INST_MODE]   = 1;
        max_component_values[INST_TARGET] = 7;
        max_component_values[INST_OP]     = 7;
        max_component_values[INST_SOURCE] = num_features - 1;
    }

    inline void randomize() {
      instruction[INST_MODE]   = Random::get<int>(0, max_component_values[INST_MODE] );
      instruction[INST_TARGET] = Random::get<int>(0, max_component_values[INST_TARGET] );
      instruction[INST_OP]     = Random::get<int>(0, max_component_values[INST_OP] );
      instruction[INST_SOURCE] = Random::get<int>(0, max_component_values[INST_SOURCE] );
    }

    inline int getMode() const {
        return instruction[INST_MODE];
    }

    inline int getTarget() const {
        return instruction[INST_TARGET];
    }

    inline int getOp() const {
        return instruction[INST_OP];
    }

    inline int getSource() const {
        return instruction[INST_SOURCE];
    }

    inline void mutate() {
        // Randomly select which component to mutate
        int component = Random::get<int>(0, INST_NUM_COMPONENTS - 1);

        // Randomly select a new value for that component
        int new_component_val = Random::get<int>(0, max_component_values[component]);
        instruction[component] = new_component_val;
    }
};

#endif //_INSTRUCTION_HPP
