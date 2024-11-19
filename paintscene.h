#pragma once
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QTimer>
#include <QDebug>

class paintScene : public QGraphicsScene
{

    Q_OBJECT

public:
    explicit paintScene(QObject* parent = 0);
    ~paintScene();

private:
    QPointF previousPoint;// The coordinates of the previous point

private:
    void mousePressEvent(QGraphicsSceneMouseEvent* event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event);
};