
// vq_gen, increase
vq_gen.up('increase', gn_balance(grid, node), sft(s, f, t))
    ${not dropVqGenInc_gn(grid, node)
      and not dropVqGenInc_gnt(grid, node, t)}
    = round(abs(ts_influx_(grid, node, f, t))+1);

// vq_gen, decrease
vq_gen.up('decrease', gn_balance(grid, node), sft(s, f, t))
    ${not dropVqGenDec_gn(grid, node)
      and not dropVqGenDec_gnt(grid, node, t)}
    = round(abs(ts_influx_(grid, node, f, t))+1);


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
    = round(abs(p_userconstraintEq_eachTimestep(group_uc, s, f, t))+1);    