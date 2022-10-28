def negamax(game, depth, alpha, beta, first=False):
    if depth == 0:
        return game.eval()
    moves = game.moves()
    if len(moves) == 0:
        return game.eval()
    # order moves
    value = None # sucks in indeterminate games.
    best_move = None
    for move in moves:
        game.move(move)
        move_value = -negamax(game, depth-1, -beta, -alpha)
        if value is None or move_value > value:
            best_move = move
            value = move_value
        game.undo_move()
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    if first:
        return value, best_move
    return value