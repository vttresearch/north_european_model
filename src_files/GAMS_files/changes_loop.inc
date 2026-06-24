// vq_userconstraintDec_t
// for each timestep, decrease, for EQ variant
vq_userconstraintDec_t.up(group_uc, sft(s, f, t))
    $ { not dropVqUserconstraint(group_uc, t)
        and sum(groupUc1234(group_uc, uc1, uc2, uc3, uc4), p_userconstraint(group_uc, uc1, uc2, uc3, uc4, 'eq'))
        and sum(groupUc1234(group_uc, uc1, uc2, uc3, uc4), p_userconstraint(group_uc, uc1, uc2, uc3, uc4, 'eachTimestep'))
        and [not group_ucSftFiltered(group_uc)
             or [group_ucSftFiltered(group_uc)
                 and sft_groupUc(group_uc, s, f, t)
                 ]
             ]
        }
    = ceil(abs(p_userconstraintEq_eachTimestep(group_uc, s, f, t)));   