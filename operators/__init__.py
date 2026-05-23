from operators.shift_time import op_shift_time

from operators.swap_random import op_swap_random

from operators.swap_charger import op_swap_charger

from operators.relocate_critical import op_relocate_critical


def build_operators(objective):

    return [

        (
            "ShiftTime",
            op_shift_time
        ),

        (
            "RelocateCrit",

            lambda assignment, instance:

            op_relocate_critical(
                assignment,
                instance,
                objective
            )
        ),

        (
            "SwapRand",
            op_swap_random
        ),

        (
            "SwapCharger",
            op_swap_charger
        ),
    ]