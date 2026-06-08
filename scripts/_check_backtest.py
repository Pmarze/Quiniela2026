import sqlite3
conn = sqlite3.connect('D:/Quiniela2026/data/quiniela.db')
conn.row_factory = sqlite3.Row

run = conn.execute('SELECT * FROM v_latest_backtest_run').fetchone()
if run:
    print('Run:', run['backtest_run_id'])

metrics = conn.execute('''
    SELECT model_id, total_quiniela_points, max_possible_points, points_efficiency,
           exact_score_accuracy, winner_accuracy, brier_1x2
    FROM v_latest_backtest_model_metrics
    ORDER BY points_efficiency DESC
''').fetchall()
print()
print(f"{'Modelo':<28} {'Pts':>5} {'Max':>5} {'Eff%':>7} {'Exact%':>7} {'Win%':>6} {'Brier':>7}")
print('-'*70)
for m in metrics:
    print(f"{m['model_id']:<28} {m['total_quiniela_points']:>5.0f} {m['max_possible_points']:>5.0f} {m['points_efficiency']*100:>7.2f} {m['exact_score_accuracy']*100:>7.2f} {m['winner_accuracy']*100:>6.2f} {m['brier_1x2']:>7.3f}")
conn.close()
