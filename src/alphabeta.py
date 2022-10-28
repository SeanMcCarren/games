def alphabeta(game, depth, alpha, beta, first=False):
    # print("alphabeta depth %d turn %d board " % (depth, game.turn) + str(game))
    if depth == 0:
        return game.eval()
    moves = game.moves()
    if len(moves) == 0:
        return game.eval()
    
    value = None # sucks in indeterminate games.
    best_move = None
    if game.turn == 0:
        #val = -inf
        for move in moves:
            game.move(move)
            move_value = alphabeta(game, depth-1, alpha, beta)
            if value is None or move_value > value:
                best_move = move
                value = move_value
            game.undo_move()
            alpha = max(alpha, value)
            if value >= beta:
                break
    else:
        #val = inf
        for move in moves:
            game.move(move)
            move_value = alphabeta(game, depth-1, alpha, beta)
            if value is None or move_value < value:
                best_move = move
                value = move_value
            game.undo_move()
            beta = min(beta, value)
            if value <= alpha:
                break

    if first:
        return value, best_move
    return value
